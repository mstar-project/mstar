"""A request admitted by ``InputIngest`` and torn down by the worker, against the real runtime.

This path had no coverage at all: every other worker test stubs
``InputIngest._add_new_request`` out. That is how eight constructor-keyword breakages
from the rid refactor stayed invisible behind a green suite.

It also pins the ownership split: the runtime mints the handle and owns the
per-request graph state, while RequestStateManager owns per_request_info.
"""
import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import torch  # noqa: F401  (import order: torch before mstar internals)
from rust_runtime import build_runtime

from mstar.communication.tensor_store import TensorStore
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.base import ShardingConfig
from mstar.graph.base import GraphNode
from mstar.model.base import WorkerGraph
from mstar.testing_utils.worker import deliver, make_worker
from mstar.utils.ipc_format import NewRequest, RemoveRequest, WorkerMessage, WorkerMessageType
from mstar.worker.node_manager_utils import RequestStateManager
from mstar.worker.stream_buffers import StreamBufferManager

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


def _worker(graphs=((WG_ID, NODE, WALK, "default"),)):
    """``graphs``: (worker graph id, node, walk, partition) per worker graph."""
    worker_graphs = [
        WorkerGraph(
            section=GraphNode(name=node, input_names={"prompt"}, outputs=[]),
            graph_walks={walk}, ranks=[0], worker_graph_id=wg_id,
        )
        for wg_id, node, walk, _ in graphs
    ]
    all_walks = {wg_id: {walk} for wg_id, _, walk, _ in graphs}
    all_nodes = {wg_id: {node} for wg_id, node, _, _ in graphs}
    all_loops = {wg_id: set() for wg_id, *_ in graphs}
    node_to_partition = {node: partition for _, node, _, partition in graphs}

    tensor_manager = _StubTensorManager()
    runtime = build_runtime(
        bookkeeping=tensor_manager.tensor_store.bookkeeping,
        my_worker_id=WORKER,
        my_worker_graphs=worker_graphs,
        all_wg_ids_to_graph_walks=all_walks,
        all_wg_ids_to_dyn_loops=all_loops,
        all_wg_ids_to_nodes=all_nodes,
        node_to_partition=node_to_partition,
        sharding_config=_sharding_config(),
    )
    request_state = RequestStateManager(node_to_partition=node_to_partition)
    w = make_worker(
        worker_id=WORKER, tensor_manager=tensor_manager, _graph_runtime=runtime,
        request_state=request_state,
        engine_manager=SimpleNamespace(
            engine=SimpleNamespace(publisher_nodes={}, skip_publish=lambda rid, keys: None),
            add_request=lambda rid, cfgs, info: None,
            remove_request=lambda rid: None,
        ),
        scheduler=SimpleNamespace(clear_rid=lambda rid: None, drop_backlogged=lambda rids: None),
        communicator=SimpleNamespace(send=lambda *a, **k: None),
        wakeup_event=SimpleNamespace(register_futures=lambda f: None),
        streams=StreamBufferManager(request_state, tensor_manager, []),
    )
    w.profile_info = SimpleNamespace(pop_request=lambda rid: None)
    return w


def _admit(w, body):
    w.ingest._dispatch([WorkerMessage(WorkerMessageType.NEW_REQUEST, body)])


def _remove(w, request_id):
    deliver(w, WorkerMessage(WorkerMessageType.REMOVE_REQUEST, RemoveRequest(request_id=request_id)))


def _new_request(request_id="r1", graph=(WG_ID, NODE, WALK, "default"), worker_graph_ids=(WG_ID,)):
    wg_id, _, walk, partition = graph
    return NewRequest(
        request_id=request_id,
        request_info=CurrentForwardPassInfo(
            request_id=request_id, graph_walk=walk, fwd_index=0,
            random_seed=0, max_tokens=1, partition_name=partition,
        ),
        partition_worker_graph_ids=[wg_id],
        worker_graph_to_workers={gid: [WORKER] for gid in worker_graph_ids},
        initial_inputs=[],
        loop_caps={},
    )


def test_admit_mints_a_handle_and_sets_up_both_sides():
    w = _worker()
    body = _new_request()
    _admit(w, body)

    rid = w._graph_runtime.get_rid_handle("r1")
    assert rid is not None, "the handle must be interned"
    assert w._graph_runtime.get_rid_string(rid) == "r1"
    # The handle is stamped onto the fwd_info so submodules can key on it.
    assert body.request_info.rid_handle == rid

    # The split: the runtime owns the graph state...
    assert w._graph_runtime.is_partition_idle(rid, "default")
    assert w._graph_runtime.get_sharding_config(rid) is not None
    # ...and RequestStateManager owns only what cannot live behind the
    # contract: the wire fwd_info and the tensor-holding stream buffers.
    assert rid in w.request_state.per_request_info
    assert w.request_state.get_fwd_info(rid, "default") is body.request_info
    assert not hasattr(w.request_state, "queues")
    assert w.tensor_manager.registered == [rid]


def test_remove_tears_down_both_sides_and_frees_the_handle():
    w = _worker()
    _admit(w, _new_request("r1"))
    rid = w._graph_runtime.get_rid_handle("r1")

    _remove(w, "r1")

    assert w._graph_runtime.get_rid_handle("r1") is None
    assert w._graph_runtime.get_sharding_config(rid) is None, \
        "the runtime must drop the request's state on removal"
    assert w._graph_runtime.get_sharding_config(rid) is None
    assert rid not in w.request_state.per_request_info
    assert w.tensor_manager.cleaned == [rid]

    # Freed handles are reused, which is why teardown has to be complete: the
    # next request gets this exact integer back.
    _admit(w, _new_request("r2"))
    assert w._graph_runtime.get_rid_handle("r2") == rid
    assert w._graph_runtime.get_rid_string(rid) == "r2"


THINKER = (0, "thinker", "thinker", "Thinker")
TALKER = (1, "talker", "talker", "Talker")


def test_a_second_partition_opens_nothing_twice():
    """Qwen3-Omni with the Thinker and the Talker on one worker gets a
    NewRequest for each, and the Talker's can land after the Thinker ran a
    step. The tensor manager and the engine open the request on the first
    only, since opening it again resets what the Thinker's steps left there."""
    w = _worker((THINKER, TALKER))
    opened = []
    w.engine_manager.add_request = lambda rid, cfgs, info: opened.append((rid, info.partition_name))

    _admit(w, _new_request("r1", THINKER, (0, 1)))
    _admit(w, _new_request("r1", TALKER, (0, 1)))

    rid = w._graph_runtime.get_rid_handle("r1")
    assert w.tensor_manager.registered == [rid]
    assert opened == [(rid, "Thinker")]
    # each partition still gets its own forward-pass info and worker graph
    assert w.request_state.get_fwd_info(rid, "Thinker").graph_walk == "thinker"
    assert w.request_state.get_fwd_info(rid, "Talker").graph_walk == "talker"
    assert all(w._graph_runtime.is_partition_idle(rid, p) for p in ("Thinker", "Talker"))

    # the next request to get the handle opens it again
    _remove(w, "r1")
    _admit(w, _new_request("r2", TALKER, (0, 1)))
    assert w._graph_runtime.get_rid_handle("r2") == rid
    assert w.tensor_manager.registered == [rid, rid]
    assert opened == [(rid, "Thinker"), (rid, "Talker")]
