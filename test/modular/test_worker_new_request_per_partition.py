"""A worker serving several partitions of a request gets one NewRequest per
partition. Only the first opens the request's tensor and resource state: a later
one can arrive after a step ran, and opening the state again resets what the
step left.

Drives the real ``Worker._add_new_request`` over the real Python graph runtime.
"""
import sys
from copy import deepcopy
from types import SimpleNamespace

sys.path.insert(0, ".")

import torch

from mstar.communication.communicator import BaseCommunicator
from mstar.communication.tensor_store import TensorStore
from mstar.communication.tensors import SharedMemoryCommunicationManager
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.base import ShardingConfig, ShardingGroup
from mstar.engine.resources import StepRunner
from mstar.engine.resources.base import KeyedWalks
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVReqConfig, KVStep, PagedKVConfig, PrefixSpan, StreamLayout
from mstar.engine.resources.kv.keys import chain
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.step import Segment, StepContext
from mstar.graph.base import GraphEdge, GraphNode, NodeAndGraphWalk
from mstar.graph.runtime.python import PythonGraphRuntime
from mstar.model.base import WorkerGraph
from mstar.utils.ipc_format import NewRequest
from mstar.worker.node_manager_utils import RequestStateManager
from mstar.worker.worker import Worker

WORKER = "w0"
RID = "req"
# partition -> (worker graph, walk); both on this worker
PARTITIONS = {"Thinker": (0, "think"), "Talker": (1, "talk")}
PAGE_SIZE = 128
KEYED_WALK = "prefill"


class _Communicator(BaseCommunicator):
    def send(self, entity_id, msg):
        pass

    def get_all_new_messages(self):
        return []


class _TensorManager:
    def __init__(self):
        self.registered = []
        self.tensor_store = TensorStore()

    def register_request(self, rid, sharding_config):
        self.registered.append(rid)

    def start_read_tensors(self, rid, edges, graph_walk=None):
        return []


class _StubTransfer:
    def __init__(self, transfer_engine_info, kv_cache, **kwargs):
        del transfer_engine_info, kv_cache, kwargs

    def get_kv_transfer_info(self, **kwargs):
        del kwargs

    def remove_request(self, request_id):
        del request_id

    def cleanup(self):
        pass


def _engine(add_request=lambda rid, cfgs: None):
    return SimpleNamespace(evictable_nodes=lambda: [], add_request=add_request)


def _worker(tensor_manager, engine_manager) -> Worker:
    graphs, walks, nodes = [], {}, {}
    for partition, (wg_id, walk) in PARTITIONS.items():
        graphs.append(WorkerGraph(
            section=GraphNode(name=partition, input_names={"x"}, outputs=[]),
            graph_walks={walk}, ranks=[0], worker_graph_id=wg_id,
        ))
        walks[wg_id], nodes[wg_id] = {walk}, {partition}
    node_to_partition = {partition: partition for partition in PARTITIONS}

    w = Worker.__new__(Worker)
    w.worker_id = WORKER
    w.tensor_manager = tensor_manager
    w.engine_manager = engine_manager
    w._graph_runtime = PythonGraphRuntime(
        my_worker_id=WORKER,
        my_worker_graphs=graphs,
        all_wg_ids_to_graph_walks=walks,
        all_wg_ids_to_dyn_loops={wg_id: set() for wg_id in walks},
        all_wg_ids_to_nodes=nodes,
        node_to_partition=node_to_partition,
        sharding_config=ShardingConfig(groups=[], tp_enabled_nodes=set(), shard_dim={}),
        tensor_manager=tensor_manager,
    )
    tensor_manager.rid_to_str = w._rid_str
    w.request_state = RequestStateManager(node_to_partition=node_to_partition)
    w.wakeup_event = SimpleNamespace(register_futures=lambda futures: None)
    w._my_consumer_connections = []
    w._draining_rids = set()
    w._last_active = {}
    w._unprocessed_messages = {}
    return w


def _new_request(partition, resource_configs=None) -> NewRequest:
    """What the conductor sends for one partition: every partition's carries
    the same configs and worker graph map."""
    wg_id, walk = PARTITIONS[partition]
    return NewRequest(
        request_id=RID,
        request_info=CurrentForwardPassInfo(
            request_id=RID, graph_walk=walk, fwd_index=0, random_seed=0, max_tokens=1,
            partition_name=partition, resource_configs=dict(resource_configs or {}),
        ),
        partition_worker_graph_ids=[wg_id],
        worker_graph_to_workers={wg: [WORKER] for wg, _ in PARTITIONS.values()},
        initial_inputs=[],
        loop_caps={},
    )


def test_only_the_first_new_request_opens_tensor_and_resource_state():
    ingested = []
    w = _worker(_TensorManager(), _engine(lambda rid, cfgs: ingested.append(rid)))

    w._add_new_request(_new_request("Thinker"))
    w._add_new_request(_new_request("Talker"))

    rid = w._rid(RID)
    assert w.tensor_manager.registered == [rid]
    assert ingested == [rid]
    # the second partition is still opened in full
    assert set(w.request_state.per_request_info[rid].per_partition_info) == set(PARTITIONS)
    for wg_id, _ in PARTITIONS.values():
        assert rid in w._graph_runtime._queues[wg_id].per_request_queues


def _kv() -> KVManager:
    kv = KVManager(
        cfg=PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=8192,
            max_num_pages=64, page_size=PAGE_SIZE,
        ),
        name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    kv.enable_prefix_cache(b"a root", walks={"main": KeyedWalks(writes=(KEYED_WALK,))})
    return kv


def _kv_config(n_tokens: int) -> KVReqConfig:
    tokens = list(range(n_tokens))
    keys = chain([tokens[at:at + PAGE_SIZE] for at in range(0, n_tokens, PAGE_SIZE)])
    return KVReqConfig(
        prefix_keys={"main": keys},
        kv_layout={"main": StreamLayout(spans=(PrefixSpan(n_tokens, n_tokens, KEYED_WALK),), keyed_spans=1)},
    )


def _kv_step(kv: KVManager, rid: int, span: int, walk: str) -> None:
    step = KVStep(segments=(Segment(rid, "main", span),))
    ctx = StepContext(request_ids=(rid,), graph_walk=walk, slot=0, capture=False)
    assert kv.admit(step, ctx).ok
    kv.plan(step, ctx)
    kv.commit(step, ctx)


def test_a_second_new_request_keeps_a_dropped_prefix_chain_dropped(monkeypatch):
    """A stream drops its chain when a write stops matching the keys. Seeding it
    again would index the next keyed write's page under keys that don't
    describe it, and a later request with that prompt would read the wrong KV."""
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)
    kv = _kv()
    runner = StepRunner({"kv": kv}, node_resources={partition: ["kv"] for partition in PARTITIONS})
    w = _worker(_TensorManager(), _engine(runner.ingest_request))
    configs = {"kv": _kv_config(2 * PAGE_SIZE)}

    w._add_new_request(_new_request("Thinker", configs))
    rid = w._rid(RID)
    # a walk the keys don't name writes the first page, so the chain ends
    _kv_step(kv, rid, PAGE_SIZE, "image")
    assert kv._streams[rid]["main"].chain is None

    w._add_new_request(_new_request("Talker", configs))
    assert kv._streams[rid]["main"].chain is None
    _kv_step(kv, rid, PAGE_SIZE, KEYED_WALK)

    assert kv._index.pages() == [], "pages were indexed under keys that don't describe them"


def _send(manager, tensor: torch.Tensor, tp_rank: int) -> GraphEdge:
    """One TP2 rank's shard of edge ``x``, stored and registered for send."""
    edge = GraphEdge(next_node="Talker", name="x", _total_fanin=2, _shard_dim=0)
    edge.tensor_info = manager.store_and_return_tensor_info(RID, {"x": [tensor]})["x"]
    for info in edge.tensor_info:
        manager.tensor_store.increment_ref(info.uuid, 1)
        info.source_tp_rank, info.source_tp_size = tp_rank, 2
    manager.register_for_send(RID, edge.tensor_info)
    return edge


def test_a_second_new_request_keeps_a_buffered_fanin_shard(tmp_path):
    """A shard that came in before the second NewRequest must still be there
    for the rest of the fan-in, or the edge waits for it forever."""
    def manager(entity):
        return SharedMemoryCommunicationManager(
            my_entity_id=entity, hostname="localhost", device="cpu",
            communicator=_Communicator(), shm_dir=str(tmp_path),
        )

    # a TP2 node on two other workers, gathered into this worker's Talker
    group = ShardingGroup(nodes={"Encoder"}, tp_size=2, graph_walks={"encode"})
    group.register_workers(["w1", "w2"], my_tp_rank=0)
    cfg = ShardingConfig(tp_enabled_nodes=set(), groups=[group], shard_dim={"x": 0})
    cfg.setup({
        NodeAndGraphWalk("Encoder", "encode"): ["w1", "w2"],
        NodeAndGraphWalk("Talker", "talk"): [WORKER],
    })
    senders = [manager("w1"), manager("w2")]
    for sender in senders:
        sender.register_request(RID, cfg)
    w = _worker(manager(WORKER), _engine())
    receiver = w.tensor_manager

    w._add_new_request(_new_request("Thinker"))
    rid = w._rid(RID)
    shards = [torch.randn(3, 4), torch.randn(3, 4)]
    edges = [_send(sender, shard, rank) for rank, (sender, shard) in enumerate(zip(senders, shards, strict=True))]
    receiver.start_read_tensors(rid, [deepcopy(edges[0])], graph_walk="talk")
    assert receiver.get_ready_tensors(graph_walk="talk").get(rid, []) == []

    w._add_new_request(_new_request("Talker"))

    receiver.start_read_tensors(rid, [deepcopy(edges[1])], graph_walk="talk")
    ready = receiver.get_ready_tensors(graph_walk="talk").get(rid, [])
    assert len(ready) == 1, "the fan-in lost its first shard"
    assert torch.equal(receiver.get_tensor(ready[0].tensor_info[0].uuid), torch.cat(shards, dim=0))
