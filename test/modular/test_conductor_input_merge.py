"""Inputs the conductor assembles from tensors persisted by different walks
(Whisper's transcript, first token from the prompt walk and the rest from the
decode loop) must reach a worker as one edge. The consumer treats a repeated
name in one forward pass as the next loop iteration's input and would run the
step with the first group only."""
from mstar.conductor.conductor import Conductor
from mstar.distributed.base import ShardingConfig
from mstar.graph.base import GraphEdge, NodeAndGraphWalk, TensorPointerInfo


def _info(uuid, walk):
    return TensorPointerInfo(
        dims=[1], dtype="int64", nbytes=8, address=0, stride=[1], uuid=uuid,
        source_session_id="s", source_entity="worker_0",
        _source_node_name="decoder", _source_graph_walk=walk,
    )


def _config():
    cfg = ShardingConfig(groups=[], tp_enabled_nodes=set(), shard_dim={}).clone_empty()
    cfg.setup({NodeAndGraphWalk("decoder", "align"): ["worker_0"]})
    return cfg


def test_groups_of_one_name_merge_in_order():
    edge = GraphEdge(next_node="decoder", name="transcript")
    edge.tensor_info = [_info("t0", "prefill_prompt"), _info("t1", "decode"), _info("t2", "decode")]
    other = GraphEdge(next_node="decoder", name="encoder_states")
    other.tensor_info = [_info("e0", "prefill")]

    per_worker = Conductor._split_inputs_to_workers(None, _config(), [edge, other], "align")

    edges = per_worker["worker_0"]
    assert [(e.name, [i.uuid for i in e.tensor_info]) for e in edges] == [
        ("transcript", ["t0", "t1", "t2"]),
        ("encoder_states", ["e0"]),
    ]
