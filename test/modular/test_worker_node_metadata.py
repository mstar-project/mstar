"""``Worker.__init__`` tells the graph runtime which of its nodes are lockstep-parallel, so
the runtime's speculation filter never speculates a transition into one.
"""

import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, ".")

pytest.importorskip("mstar_rust")

from rust_runtime import build_runtime  # noqa: E402

from mstar.distributed.base import ShardingConfig  # noqa: E402
from mstar.graph.base import GraphEdge, GraphNode, Loop, Sequential  # noqa: E402
from mstar.graph.runtime.base import ColumnarEdgeSpecs  # noqa: E402
from mstar.graph.runtime.rust import RustGraphRuntime  # noqa: E402
from mstar.model.base import WorkerGraph  # noqa: E402
from mstar.utils.containers import ParallelList  # noqa: E402
from mstar.worker import worker as worker_mod  # noqa: E402

WALK = "decode"
NODES = {"prefill", "ar_decode"}


def _runtime_at_prefill() -> tuple[RustGraphRuntime, int]:
    """A runtime holding one request whose prefill is ready, feeding ar_decode's loop."""
    loop_edges = [GraphEdge(name="token", next_node="ar_decode")]
    graph = Sequential(sections=[
        GraphNode(name="prefill", input_names={"prompt"}, outputs=loop_edges),
        Loop(
            name="ar_loop",
            section=GraphNode(name="ar_decode", input_names={"token"}, outputs=loop_edges),
            outputs=[GraphEdge(name="token", next_node="post_processor")],
            max_iters=4,
        ),
    ])
    runtime = build_runtime(
        my_worker_id="worker0",
        my_worker_graphs=[WorkerGraph(section=graph, graph_walks={WALK}, ranks=[0], worker_graph_id=0)],
        all_wg_ids_to_graph_walks={0: {WALK}},
        all_wg_ids_to_dyn_loops={0: {"ar_loop"}},
        all_wg_ids_to_nodes={0: NODES},
        node_to_partition=dict.fromkeys(NODES, "default"),
        sharding_config=ShardingConfig(groups=[], tp_enabled_nodes=set(), shard_dim={}),
    )
    rid = runtime.add_request(
        request_id="r", partition="default", graph_walk=WALK, partition_worker_graph_ids=[0],
        worker_graph_to_workers=ParallelList([0], [["worker0"]]), loop_caps={},
    )
    block = ColumnarEdgeSpecs.empty()
    block.add_edge(rid, GraphEdge(name="prompt", next_node="prefill"))
    runtime.ingest_inputs_batch(block, can_buffer=True)
    return runtime, rid


def _init_worker(monkeypatch, runtime: RustGraphRuntime, parallel: set[str]) -> None:
    """Run ``Worker.__init__`` over stubs, with ``runtime`` as its graph runtime."""
    monkeypatch.delenv("MSTAR_TP_ASYNC_SCHED", raising=False)
    monkeypatch.setattr(worker_mod, "make_communicator", lambda **kw: SimpleNamespace(
        register_event_for_poll=lambda event: None))
    monkeypatch.setattr(worker_mod, "EventWakeup", object)
    monkeypatch.setattr(worker_mod, "create_tensor_communication_manager", lambda **kw: SimpleNamespace(
        transfer_engine=None, my_session_id="s", get_tensor=None))
    monkeypatch.setattr(worker_mod, "_make_graph_runtime", lambda **kw: runtime)
    # an engine per node that declares no stop policies
    engine = SimpleNamespace(get_stop_policies=lambda node, walk: None)
    monkeypatch.setattr(worker_mod, "EngineManager", SimpleNamespace(
        build=lambda *a, **kw: SimpleNamespace(get_engine=lambda node: engine, side_keys=())))
    groups = SimpleNamespace(
        init_dist=lambda **kw: None,
        get_instance_rank_for_node=lambda node: 0,
        get_instance_world_size_for_node=lambda node: 2 if node in parallel else 1,
    )
    worker_mod.Worker(
        worker_id="worker0", worker_ids=["worker0"], model=None,
        my_worker_graphs=[SimpleNamespace(
            section=SimpleNamespace(get_nodes=lambda: NODES, get_loops=lambda: {}), graph_walks={WALK})],
        model_config={}, all_worker_graph_ids_to_graph_walks={},
        all_worker_graph_ids_to_nodes={}, all_worker_graph_ids_to_dyn_loops={},
        sharding_config=None, parallel_groups=groups, device=torch.device("cpu"),
    )


def test_a_transition_into_a_parallel_node_is_not_speculated(monkeypatch):
    runtime, rid = _runtime_at_prefill()
    assert [o.node_name for o in runtime.speculate_node("prefill", [WALK], [rid])] == ["ar_decode"]

    _init_worker(monkeypatch, runtime, parallel={"ar_decode"})

    assert runtime.speculate_node("prefill", [WALK], [rid]) == []
