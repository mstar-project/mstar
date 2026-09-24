"""Worker._add_new_request / _remove_request against the real runtime.

This path had no coverage at all: every other worker test stubs
``_add_new_request`` out. That is how eight constructor-keyword breakages
from the rid refactor stayed invisible behind a green suite.

It also pins the ownership split the graph-runtime port introduced: the
runtime mints the handle and owns the per-request QUEUE lifecycle, while
RequestStateManager owns per_request_info and shares the same queues dict.
"""
import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import torch  # noqa: F401  (import order: torch before mstar internals)

from mstar.communication.tensor_store import TensorStore
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.base import ShardingConfig
from mstar.graph.base import GraphNode
from mstar.graph.runtime.python import PythonGraphRuntime
from mstar.model.base import WorkerGraph
from mstar.utils.ipc_format import NewRequest, RemoveRequest
from mstar.worker.node_manager_utils import RequestStateManager
from mstar.worker.worker import Worker

WG_ID = 0
WALK = "prefill"
NODE = "prefill"
WORKER = "worker0"


class _StubTensorManager:
    def __init__(self):
        self.registered = []
        self.cleaned = []
        # The runtime resolves uuids to descriptors through this.
        self.tensor_store = TensorStore()

    def register_request(self, rid, sharding_config):
        self.registered.append(rid)

    def start_read_tensors(self, rid, edges, graph_walk=None):
        return []

    def force_cleanup_request(self, rid):
        self.cleaned.append(rid)


def _sharding_config():
    """Un-sharded base config; add_request clones it and calls setup."""
    return ShardingConfig(groups=[], tp_enabled_nodes=set(), shard_dim={})


def _worker():
    section = GraphNode(name=NODE, input_names={"prompt"}, outputs=[])
    worker_graph = WorkerGraph(
        section=section, graph_walks={WALK}, ranks=[0], worker_graph_id=WG_ID,
    )
    all_walks = {WG_ID: {WALK}}
    all_nodes = {WG_ID: {NODE}}
    all_loops = {WG_ID: set()}
    node_to_partition = {NODE: "default"}

    w = Worker.__new__(Worker)
    w.worker_id = WORKER
    w.is_tp_follower = False
    w.tensor_manager = _StubTensorManager()
    w._graph_runtime = PythonGraphRuntime(
        my_worker_id=WORKER,
        my_worker_graphs=[worker_graph],
        all_wg_ids_to_graph_walks=all_walks,
        all_wg_ids_to_dyn_loops=all_loops,
        all_wg_ids_to_nodes=all_nodes,
        node_to_partition=node_to_partition,
        sharding_config=_sharding_config(),
        tensor_manager=w.tensor_manager,
    )
    w.request_state = RequestStateManager(node_to_partition=node_to_partition)
    w.engine_manager = SimpleNamespace(
        evictable_nodes=lambda: [NODE],
        add_request=lambda rid, cfgs: None,
        remove_request=lambda rid: None,
    )
    w.scheduler = SimpleNamespace(clear_rid=lambda rid: None)
    w.profile_info = SimpleNamespace(pop_request=lambda rid: None)
    w.communicator = SimpleNamespace(send=lambda *a, **k: None)
    w.wakeup_event = SimpleNamespace(register_futures=lambda f: None)
    w._my_consumer_connections = []
    w._draining_rids = set()
    w._pending_drains = set()
    w._reads_done_sent = set()
    w._in_flight_rids = set()
    w._pending_removes = set()
    w._last_active = {}
    w.streaming_buffers = {}
    w._unprocessed_messages = {}
    w.enable_nvtx = False
    return w


def _new_request(request_id="r1"):
    return NewRequest(
        request_id=request_id,
        request_info=CurrentForwardPassInfo(
            request_id=request_id, graph_walk=WALK, fwd_index=0,
            random_seed=0, max_tokens=1, partition_name="default",
        ),
        partition_worker_graph_ids=[WG_ID],
        worker_graph_to_workers={WG_ID: [WORKER]},
        initial_inputs=[],
    )


def test_admit_mints_a_handle_and_sets_up_both_sides():
    w = _worker()
    body = _new_request()
    w._add_new_request(body)

    rid = w._rid("r1")
    assert rid is not None, "the handle must be interned"
    assert w._rid_str(rid) == "r1"
    # The handle is stamped onto the fwd_info so submodules can key on it.
    assert body.request_info.rid_handle == rid

    # The split: the runtime owns the graph state...
    assert rid in w._graph_runtime._queues[WG_ID].per_request_queues
    assert w._graph_runtime.get_sharding_config(rid) is not None
    # ...and RequestStateManager owns only what cannot live behind the
    # contract: the wire fwd_info and the tensor-holding stream buffers.
    assert rid in w.request_state.per_request_info
    assert w.request_state.get_fwd_info(rid, "default") is body.request_info
    assert not hasattr(w.request_state, "queues")
    assert w.tensor_manager.registered == [rid]


def test_remove_tears_down_both_sides_and_frees_the_handle():
    w = _worker()
    w._add_new_request(_new_request("r1"))
    rid = w._rid("r1")

    w._remove_request(RemoveRequest(request_id="r1"))

    assert w._rid("r1") is None
    assert rid not in w._graph_runtime._queues[WG_ID].per_request_queues, \
        "the runtime must drop the per-request queue on removal"
    assert w._graph_runtime.get_sharding_config(rid) is None
    assert rid not in w.request_state.per_request_info
    assert w.tensor_manager.cleaned == [rid]

    # Freed handles are reused, which is why teardown has to be complete: the
    # next request gets this exact integer back.
    w._add_new_request(_new_request("r2"))
    assert w._rid("r2") == rid
    assert w._rid_str(rid) == "r2"


def test_admit_is_refused_after_a_drain():
    """A NEW can arrive after the DRAIN that tears the rid down; admitting it
    would leave state behind that no REMOVE will ever collect."""
    w = _worker()
    w._draining_rids.add("r1")
    w._add_new_request(_new_request("r1"))
    assert w._rid("r1") is None
    assert w.tensor_manager.registered == []
