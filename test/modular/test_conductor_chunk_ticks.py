"""The conductor seeds a chunk Loop's loop-back on the ranks hosting the chunked
node. A walk's other inputs can all go to a different rank (Qwen3-Omni's
``prefill_audio`` sends every edge to the audio encoder), and a seed that only
reached those ranks would leave the Thinker's Loop unseeded."""
from types import SimpleNamespace

from mstar.conductor.conductor import Conductor
from mstar.conductor.request_info import CurrentForwardConductorMetadata
from mstar.distributed.base import ShardingConfig
from mstar.graph.base import GraphEdge, NodeAndGraphWalk, TensorPointerInfo
from mstar.model.base import ForwardPassArgs
from mstar.model.submodule_base import chunk_tick_name

WALK = "prefill_audio"
TICK = chunk_tick_name("Thinker", WALK)


def _info(uuid):
    return TensorPointerInfo(
        dims=[1], dtype="float32", nbytes=4, address=0, stride=[1], uuid=uuid,
        source_session_id="s", source_entity="conductor",
        _source_node_name="conductor", _source_graph_walk=WALK,
    )


def _conductor():
    cfg = ShardingConfig(groups=[], tp_enabled_nodes=set(), shard_dim={}).clone_empty()
    cfg.setup({
        NodeAndGraphWalk("audio_encoder", WALK): ["worker_0"],
        NodeAndGraphWalk("Thinker", WALK): ["worker_1"],
    })
    c = Conductor.__new__(Conductor)
    c._chunk_ticks = {WALK: (("Thinker", WALK),)}
    pstate = SimpleNamespace(fwd_pass_number=0, random_seed=0, resource_publish_info=None)
    c.requests = {"r": SimpleNamespace(
        sharding_config=cfg, partition_states={"Thinker": pstate},
        max_output_tokens=8, resource_configs={},
    )}
    c._update_persist_ref_counts = lambda rid, edges: None
    sent = {}
    c.communicator = SimpleNamespace(send=lambda w, m: sent.setdefault(w, []).extend(m.body.inputs))
    return c, sent


def _audio_inputs():
    edge = GraphEdge(next_node="audio_encoder", name="input_features")
    edge.tensor_info = [_info("a0")]
    return [edge]


def test_tick_reaches_the_chunked_nodes_rank_not_the_encoders():
    c, sent = _conductor()
    fwd = ForwardPassArgs(
        full_metadata=CurrentForwardConductorMetadata(graph_walk=WALK, is_prefill=True),
        inputs=_audio_inputs(), unpersist_tensors=[],
    )
    c._send_partition_inputs("r", "Thinker", fwd)
    assert [e.name for e in sent["worker_0"]] == ["input_features"]
    assert [e.name for e in sent["worker_1"]] == [TICK]


def test_a_client_stream_chunk_carries_no_tick():
    c, sent = _conductor()
    c._send_inputs("r", "Thinker", _audio_inputs(), WALK, {})
    assert [e.name for e in sent["worker_0"]] == ["input_features"]
    assert "worker_1" not in sent
