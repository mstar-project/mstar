"""The conductor seeds a chunk Loop's loop-back on the ranks hosting the chunked
node. A walk's other inputs can all go to a different rank (Qwen3-Omni's
``prefill_audio`` sends every edge to the audio encoder), and a seed that only
reached those ranks would leave the Thinker's Loop unseeded. The seed is the
walk plan's own (rule 1: a loop-back input with nothing held goes empty), so it
is routed like any other edge, by the node it names."""
from types import SimpleNamespace

from test_ingest_unpersist import _ToyModel, plan_step

from mstar.conductor.conductor import Conductor
from mstar.conductor.request_info import PartitionState, WalkStep
from mstar.distributed.base import ShardingConfig
from mstar.graph.base import GraphEdge, GraphNode, NodeAndGraphWalk, Sequential, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.submodule_base import chunk_tick_name

WALK = "prefill_audio"
TICK = chunk_tick_name("llm", WALK)


def _info(uuid):
    return TensorPointerInfo(
        dims=[1], dtype="float32", nbytes=4, address=0, stride=[1], uuid=uuid,
        source_session_id="s", source_entity="conductor",
        _source_node_name="conductor", _source_graph_walk=WALK,
    )


def test_tick_reaches_the_chunked_nodes_rank_not_the_encoders():
    model = _ToyModel({WALK: Sequential([
        GraphNode(name="audio_encoder", input_names=["input_features"],
                  outputs=[GraphEdge(next_node="llm", name="audio_embeds")]),
        GraphNode(name="llm", input_names=["audio_embeds"],
                  outputs=[GraphEdge(next_node=EMIT_TO_CLIENT, name="new_token", output_modality="text")]),
    ])}, {"default": [WalkStep(WALK)]}, chunked=(WALK,))
    step = plan_step(model, [WalkStep(WALK)], 0, {"input_features": [_info("a0")]})

    cfg = ShardingConfig(groups=[], tp_enabled_nodes=set(), shard_dim={}).clone_empty()
    cfg.setup({
        NodeAndGraphWalk("audio_encoder", WALK): ["worker_0"],
        NodeAndGraphWalk("llm", WALK): ["worker_1"],
    })
    c = Conductor.__new__(Conductor)
    c.requests = {"r": SimpleNamespace(
        sharding_config=cfg, partition_states={"default": PartitionState("default")},
        max_output_tokens=8, resource_configs={},
    )}
    c._update_persist_ref_counts = lambda rid, edges: None
    sent = {}
    c.communicator = SimpleNamespace(send=lambda w, m: sent.setdefault(w, []).extend(m.body.inputs))
    c._send_inputs("r", "default", step.inputs, WALK, step.step_metadata)

    assert [e.name for e in sent["worker_0"]] == ["input_features"]
    assert [e.name for e in sent["worker_1"]] == [TICK]
