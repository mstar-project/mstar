"""A walk the cache holds whole runs nothing, and its node still completes.

When a hit covers every token a walk would write, the walk has no forward to
run, but the graph still waits on its node: the conductor moves a request to
its next walk only once every worker graph reports done. A rid left out of the
step the way a vetoed one is would also be left out of routing, and the
request would sit there until the client gave up.

A text walk is keyed by its ids, whose positions are their count, so one
placing positions of its own fails, probed or refused: its pages would be
served later at positions they never had.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch

from mstar.communication.tensors import StoredOutputs
from mstar.engine.engine import Engine, ExecutingBatch
from mstar.engine.resources import StepContext, StepRunner
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVReqConfig, KVSpec, PagedKVConfig, PrefixSpan
from mstar.engine.resources.kv.keys import chain
from mstar.engine.resources.kv.manager import KVManager
from mstar.graph.runtime.base import FreedTensors, RouteOutput
from mstar.model.base import PrefixStream
from mstar.model.submodule_base import ARNodeInputs, BatchedModelOutput
from mstar.worker.worker import PendingBatch, Worker

RID = 7
NODE = "LLM"
WALK = "prefill"


def _batch(rids=(RID,)) -> ExecutingBatch:
    return ExecutingBatch(
        node_name=NODE,
        per_request_info={rid: None for rid in rids},
        step_context=StepContext(
            request_ids=tuple(rids), graph_walk=WALK, slot=0, capture=False,
        ),
    )


class _StubGraphRuntime:
    """Records what it was asked to complete, and routes nothing onward."""

    def __init__(self):
        self.routed = None

    def cleanup_consumed_inputs(self, node_name, rids, wg_ids):
        return FreedTensors.none()

    def clear_pending_loop_stops(self):
        pass

    def get_dynamic_loop_iters(self, rids, partition):
        return []

    def complete_and_route_batch(self, route_input):
        self.routed = route_input
        return RouteOutput(
            completion_id=0, register_uuids=[], register_rids=[],
            new_token_output_idxs=[], local_streaming_by_signal={},
        )

    def send_outputs(self, send_input):
        return []


class _StubTensorManager:
    """Counts each rid's outputs and stores none of them."""

    def cleanup_collectable(self, *freed):
        pass

    def store_and_return_tensor_info_batch(self, rids, outputs, signals, **kwargs):
        return StoredOutputs(
            flat_uuids=[], flat_rids=[], signal_idxs=[],
            num_tensors=[
                len((outputs.get(rid) or {}).get(signal, ()))
                for rid in rids for signal in signals
            ],
        )

    def increment_ref_batch_uniform(self, uuids, count):
        pass

    def dereference_batch_uniform(self, uuids):
        pass


def _worker() -> Worker:
    w = Worker.__new__(Worker)
    w._graph_runtime = _StubGraphRuntime()
    w.tensor_manager = _StubTensorManager()
    w.engine_manager = SimpleNamespace(get_engine=lambda node: SimpleNamespace(
        check_stop_for_batch=lambda batch, outputs, **kwargs: {},
        extend_prefix_chains=lambda batch, outputs: None,
    ))
    w.request_state = SimpleNamespace(
        per_request_info={}, get_fwd_info=lambda rid, partition: None,
        buffer_publish_info=lambda rid, partition, published: None,
        get_pending_publish_info=lambda rid, partition: {},
    )
    w.device = torch.device("cpu")
    w.enable_nvtx = False
    w.enable_prof = False
    w._phase_period = 0
    w._last_active = {}
    w._innermost_loop_by_node = {}
    return w


def test_a_served_walk_is_routed_with_no_tensors():
    worker = _worker()
    node_batch = _batch(rids=())
    node_batch.cached_rids = {RID}
    pending = PendingBatch(
        batch=SimpleNamespace(
            request_to_worker_graph={RID: 0}, node_name=NODE,
            output_signals=["new_token"],
        ),
        node_batch=node_batch, node_name=NODE, partition="default",
        graph_walk=WALK, future=None,
    )

    worker._postprocess_batch(pending, BatchedModelOutput())

    routed = worker._graph_runtime.routed
    assert routed is not None and list(routed.wg_ids.keys) == [RID], (
        "a walk the cache served was never routed, so its node never "
        "completes and the request never reaches its next walk"
    )
    assert routed.num_tensors == [0], "a served walk routed tensors it never produced"


@pytest.mark.parametrize("refused", [False, True], ids=["probed", "refused"])
def test_a_keyed_text_walk_placing_its_own_positions_fails_probed_or_refused(monkeypatch, refused):
    monkeypatch.setattr(manager_mod, "KVTransferManager", lambda *args, **kwargs: None)
    kv = KVManager(
        cfg=PagedKVConfig(num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=4096, max_num_pages=16),
        name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    kv.enable_prefix_cache(b"a root", {"main": (WALK, "decode")})
    kv.ingest_request(RID, KVReqConfig(
        prefix_keys={"main": chain([list(range(100))])}, prefix_layout={"main": [PrefixSpan(100, 100, WALK)]},
    ))
    engine = Engine.__new__(Engine)
    engine._runner = StepRunner({"kv": kv}, node_resources={NODE: ["kv"]})
    engine._keyed_walks = {NODE: {WALK}}
    engine._prefix_model = "_Model"
    inputs = ARNodeInputs(
        input_ids=torch.arange(100), input_seq_len=100, custom_pos_ids=torch.arange(100), resource_step_info=refused,
    )

    with pytest.raises(AssertionError, match="the layout keys it by ids"):
        engine._skip_cached_prefix(_batch(), RID, inputs)


class _StubModel:
    """Declares a stream an image walk writes too, and names no checkpoint."""

    def checkpoint_path(self):
        return None

    def preprocess_fingerprint(self):
        return "stub"

    def prefix_key_streams(self):
        return {"kv": {"main": PrefixStream("text_inputs", "ids", WALK, "decode", ("prefill_image",))}}


def test_the_layout_walks_reach_both_the_probe_and_the_cache(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", lambda *args, **kwargs: None)
    config = PagedKVConfig(num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=64, prefix_cache_salt="a salt")
    kv = KVManager(
        cfg=config, name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    engine = Engine.__new__(Engine)
    engine._resources = {"kv": kv}

    engine._open_prefix_caches({"kv": KVSpec(resource_key="kv", nodes={NODE}, config=config)}, _StubModel())

    walks = (engine._keyed_walks[NODE], kv._keyed_walks)
    assert walks == ({WALK, "prefill_image"}, {"main": {WALK, "decode", "prefill_image"}}), (
        "the engine never probes the image walk, or the cache ends the chain at its first commit"
    )
