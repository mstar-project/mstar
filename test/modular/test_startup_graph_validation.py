"""Graph, partition and node_groups declarations the conductor cannot serve
are rejected when it starts, not served wrongly later.

``validate_deployment`` refuses: a walk in no partition or in two, a
partition naming an undefined walk, a node placed by no node_group or by two
for one walk, a node_group naming an undefined walk, and the stream-driven
shapes the runtime and conductor can't account for: a self-triggered consumer
(every input streamed) sharing its worker graph with another top-level entity,
and a consumer walk across worker graphs, unless its consumers are gated and
its worker graphs share workers. The sweep at the bottom builds every shipped config
through ``Conductor.__init__`` (workers and sockets stubbed, no weights) so
none of them trips the new checks.
"""
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
import yaml
from test_graph_runtime import (
    StubTensorManager,
    _fwd_info,
    _ingest,
    _RecordingCommunicator,
    _sharding_config,
    _store_outputs,
)

import mstar.conductor.conductor as conductor_mod
from mstar.communication.tensor_uuid import TensorUuidMinter
from mstar.conductor.conductor import Conductor
from mstar.conductor.request_info import PartitionDefinition
from mstar.distributed.base import ShardingConfig
from mstar.graph.base import GraphEdge, GraphNode, Loop, Sequential, TensorPointerInfo
from mstar.graph.runtime.base import ColumnarEdgeSpecs, RouteInput, SendInput
from mstar.graph.runtime.python import PythonGraphRuntime
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
from mstar.model.base import Model
from mstar.model.registry import MODEL_REGISTRY, get_model_class, model_init_kwargs
from mstar.streaming.chunk_policy import FixedChunkPolicy
from mstar.streaming.topology import Connection, PartitionTopology, StreamingGraphEdge
from mstar.utils.containers import ParallelList
from mstar.validation import DeploymentError, validate_deployment

ROOT = Path(__file__).resolve().parents[2]


class _ToyModel(Model):
    def __init__(self, walks, partitions=None, topology=None, tp_enabled_nodes=()):
        self._walks = walks
        self._partitions = partitions
        self._topology = topology
        self._tp_enabled_nodes = set(tp_enabled_nodes)

    def get_graph_walk_graphs(self):
        return self._walks()

    def get_partitions(self):
        return self._partitions() if self._partitions else super().get_partitions()

    def get_partition_topology(self):
        return self._topology() if self._topology else super().get_partition_topology()

    def get_default_sharding_config(self):
        return ShardingConfig(groups=[], tp_enabled_nodes=self._tp_enabled_nodes, shard_dim={})

    def get_node_resources(self):
        return []

    def get_initial_forward_pass_args(self, *a, **k):
        raise NotImplementedError

    def get_partition_forward_pass_args(self, *a, **k):
        raise NotImplementedError

    def process_prompt(self, *a, **k):
        return {}

    def postprocess(self, *a, **k):
        return b""

    def get_submodule(self, *a, **k):
        return None


def _config(tmp_path, node_groups) -> str:
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({"max_seq_len": 64, "node_groups": node_groups}))
    return str(path)


def _validate(model, config_path):
    return validate_deployment(model, yaml.safe_load(Path(config_path).read_text()), config_path)


def _start_conductor(model, config_path, tmp_path, monkeypatch) -> Conductor:
    monkeypatch.setattr(Conductor, "_launch_workers", lambda self: None)
    monkeypatch.setattr(conductor_mod, "make_communicator", lambda **kw: None)
    return Conductor(model, config_path, socket_path_prefix=str(tmp_path / "s"))


def _llm(name="llm"):
    return GraphNode(name=name, input_names=["x"], outputs=[
        GraphEdge(next_node=EMIT_TO_CLIENT, name="y", output_modality="text"),
    ])


def _prefill_decode():
    return {"prefill": _llm(), "decode": _llm()}


# -- partitions ---------------------------------------------------------------


def test_walk_in_no_partition_is_rejected(tmp_path, monkeypatch):
    """On main the walk is filed under 'default', its worker graph is never
    registered on a worker, and the conductor still routes to it: a hang."""
    model = _ToyModel(_prefill_decode, partitions=lambda: [
        PartitionDefinition(name="default", graph_walks={"prefill"}),
    ])
    cfg = _config(tmp_path, [{"node_names": ["llm"], "ranks": [0]}])
    with pytest.raises(ValueError, match=r"Graph walk 'decode' is in 0 partitions \[\]"):
        _start_conductor(model, cfg, tmp_path, monkeypatch)


def test_walk_in_two_partitions_is_rejected(tmp_path):
    model = _ToyModel(_prefill_decode, partitions=lambda: [
        PartitionDefinition(name="A", graph_walks={"prefill", "decode"}),
        PartitionDefinition(name="B", graph_walks={"decode"}),
    ])
    cfg = _config(tmp_path, [{"node_names": ["llm"], "ranks": [0]}])
    with pytest.raises(ValueError, match=r"Graph walk 'decode' is in 2 partitions \['A', 'B'\]"):
        _validate(model, cfg)


def test_partition_naming_an_undefined_walk_is_rejected(tmp_path):
    model = _ToyModel(_prefill_decode, partitions=lambda: [
        PartitionDefinition(name="default", graph_walks={"prefill", "decode", "decod"}),
    ])
    cfg = _config(tmp_path, [{"node_names": ["llm"], "ranks": [0]}])
    with pytest.raises(ValueError, match=r"Partition 'default' lists graph walks \['decod'\]"):
        _validate(model, cfg)


# -- node_groups --------------------------------------------------------------


def test_node_in_two_node_groups_is_rejected(tmp_path, monkeypatch):
    """On main the later group silently wins: worker_0 is never launched."""
    cfg = _config(tmp_path, [
        {"node_names": ["llm"], "ranks": [0]},
        {"node_names": ["llm"], "ranks": [1]},
    ])
    with pytest.raises(ValueError, match=(
        r"node 'llm' of graph walk '(prefill|decode)' is placed by node_groups\[0\], node_groups\[1\]"
    )):
        _start_conductor(_ToyModel(_prefill_decode), cfg, tmp_path, monkeypatch)


def test_node_in_no_node_group_is_rejected(tmp_path):
    cfg = _config(tmp_path, [{"node_names": ["llm"], "ranks": [0], "graph_walks": ["prefill"]}])
    with pytest.raises(ValueError, match=r"node 'llm' of graph walk 'decode' is placed by no node_groups entry"):
        _validate(_ToyModel(_prefill_decode), cfg)


def test_node_group_naming_an_undefined_walk_is_rejected(tmp_path):
    cfg = _config(tmp_path, [
        {"node_names": ["llm"], "ranks": [0], "graph_walks": ["prefill"]},
        {"node_names": ["llm"], "ranks": [1], "graph_walks": ["decode", "deocde"]},
    ])
    with pytest.raises(ValueError, match=r"node_groups\[1\] \(nodes \['llm'\]\) lists graph_walks \['deocde'\]"):
        _validate(_ToyModel(_prefill_decode), cfg)


def test_one_node_on_different_ranks_per_walk_is_accepted(tmp_path, monkeypatch):
    cfg = _config(tmp_path, [
        {"node_names": ["llm"], "ranks": [0], "graph_walks": ["prefill"]},
        {"node_names": ["llm"], "ranks": [1], "graph_walks": ["decode"]},
    ])
    c = _start_conductor(_ToyModel(_prefill_decode), cfg, tmp_path, monkeypatch)
    assert c.node_walk_to_wg[("llm", "prefill")].ranks == [0]
    assert c.node_walk_to_wg[("llm", "decode")].ranks == [1]


# -- stream-driven walks ------------------------------------------------------


def _stream_model(chunk_walk, extra_walks=None, tp_enabled_nodes=(), producer=None):
    """Producer partition P streams ``codes`` into consumer partition C."""
    def walks():
        out = {
            "gen": producer() if producer else GraphNode(name="producer", input_names=["prompt"], outputs=[
                StreamingGraphEdge(next_node="vocoder", name="codes", target_partition="C"),
            ]),
            "chunk": chunk_walk(),
        }
        out.update({k: v() for k, v in (extra_walks or {}).items()})
        return out

    return _ToyModel(
        walks,
        partitions=lambda: [
            PartitionDefinition(name="P", graph_walks={"gen"}, initial_walk="gen"),
            PartitionDefinition(name="C", graph_walks={"chunk", *(extra_walks or {})},
                                initial_walk="chunk", producer_partitions=["P"]),
        ],
        topology=lambda: PartitionTopology(partitions=["P", "C"], connections=[
            Connection(from_partition="P", to_partition="C", edge_name="codes",
                       chunk_policy_factory=lambda: FixedChunkPolicy(chunk_size=1)),
        ]),
        tp_enabled_nodes=tp_enabled_nodes,
    )


def _vocoder(outputs=None):
    return GraphNode(name="vocoder", input_names=["codes"], outputs=outputs or [
        GraphEdge(next_node=EMIT_TO_CLIENT, name="audio", output_modality="audio"),
    ])


def _vocoder_then_post():
    return Sequential([
        _vocoder([GraphEdge(next_node="post", name="wave")]),
        GraphNode(name="post", input_names=["wave"], outputs=[
            GraphEdge(next_node=EMIT_TO_CLIENT, name="audio", output_modality="audio"),
        ]),
    ])


def _gated_vocoder_then_post():
    """The vocoder also waits on a ``trigger`` the conductor sends each pass."""
    walk = _vocoder_then_post()
    walk.sections[0] = GraphNode(name="vocoder", input_names=["codes", "trigger"],
                                 outputs=[GraphEdge(next_node="post", name="wave")])
    return walk


def test_self_triggered_consumer_sharing_its_worker_graph_is_rejected(tmp_path):
    """A consumer and a same-group successor merge into one worker graph. Its
    completion counts entity runs (the re-armed consumer running twice reads
    as done) and the pass-end reset drops the successor's buffered input."""
    cfg = _config(tmp_path, [{"node_names": ["producer", "vocoder", "post"], "ranks": [0]}])
    with pytest.raises(ValueError, match=(
        r"graph walk 'chunk' of consumer partition 'C' has self-triggered consumers \['vocoder'\] "
        r"\(every input streamed\) among 2 top-level entities \['post', 'vocoder'\] in one worker graph"
    )):
        _validate(_stream_model(_vocoder_then_post), cfg)


def _loop_then_post():
    return Sequential([
        Loop(name="chunk_loop", max_iters=4, outputs=[], section=_vocoder([GraphEdge(next_node="post", name="wave")])),
        GraphNode(name="post", input_names=["wave"], outputs=[
            GraphEdge(next_node=EMIT_TO_CLIENT, name="audio", output_modality="audio"),
        ]),
    ])


def test_self_triggered_consumer_loop_sharing_its_worker_graph_is_rejected(tmp_path):
    """A finished Loop reseeds the streaming readiness of a member whose every
    input is streamed, so the Loop restarts on the next chunk."""
    cfg = _config(tmp_path, [{"node_names": ["producer", "vocoder", "post"], "ranks": [0]}])
    with pytest.raises(ValueError, match=r"has self-triggered consumers \['chunk_loop'\]"):
        _validate(_stream_model(_loop_then_post), cfg)


class _ChunkWalkRuntime:
    """The Python graph runtime of worker_0, hosting the 'chunk' walk's worker
    graphs, driven the way the worker drives it."""

    def __init__(self, node_groups, tmp_path):
        wgs = _stream_model(_gated_vocoder_then_post).get_worker_graphs(_config(tmp_path, node_groups))
        for i, wg in enumerate(wgs):
            wg.worker_graph_id = i
        self.wgs = [wg for wg in wgs if "chunk" in wg.graph_walks]
        self.wg_of = {node: wg.worker_graph_id for wg in self.wgs for node in wg.section.get_nodes()}
        ids = [wg.worker_graph_id for wg in self.wgs]
        self.rt = PythonGraphRuntime(
            my_worker_id="worker_0", my_worker_graphs=self.wgs,
            all_wg_ids_to_graph_walks={i: {"chunk"} for i in ids},
            all_wg_ids_to_dyn_loops={i: set() for i in ids},
            all_wg_ids_to_nodes={wg.worker_graph_id: set(wg.section.get_nodes()) for wg in self.wgs},
            node_to_partition={"vocoder": "C", "post": "C"}, sharding_config=_sharding_config(),
            tensor_manager=StubTensorManager(),
        )
        self.rt._communicator = _RecordingCommunicator()
        self.rid = self.rt.add_request(
            request_id="r", partition="C", graph_walk="chunk", partition_worker_graph_ids=ids,
            worker_graph_to_workers=ParallelList(ids, [["worker_0"]] * len(ids)),
            loop_caps={},
        )
        self.minter = TensorUuidMinter("worker_0")

    def trigger(self):
        _ingest(self.rt, self.rid, [GraphEdge(next_node="vocoder", name="trigger")])

    def feed_chunk(self):
        """What the worker's stream poll does with a popped chunk; returns the refused edges."""
        block = ColumnarEdgeSpecs.empty()
        block.add_edge(self.rid, GraphEdge(name="codes", next_node="vocoder", is_streaming=True))
        return self.rt.ingest_inputs_batch(block, can_buffer=False, is_streaming=True)

    def run(self, node):
        """Run one step of ``node``; returns the worker graphs it completed."""
        rt, rid = self.rt, self.rid
        sig = rt.get_output_signals(node, "chunk")
        uuids = _store_outputs(rt._tensor_store, self.minter, rid, {s: [torch.ones(1)] for s in sig})
        assert rt.pop_rids(node, "chunk", [rid], check_ready=True) is not None, f"{node} not ready"
        out = rt.complete_and_route_batch(RouteInput(
            partition="C", graph_walk="chunk", node_name=node, output_signals=sig,
            wg_ids=ParallelList([rid], [self.wg_of[node]]),
            tensors=[u for s in sig for u in uuids[s]], num_tensors=[1] * len(sig),
        ))
        done = rt._peek_completion(out.completion_id).routing[rid].completed_worker_graph_ids
        rt.send_outputs(SendInput(
            completion_id=out.completion_id, new_token_counts=ParallelList([rid], [{}]),
            per_request_info=ParallelList([rid], [_fwd_info("chunk", "C")]),
        ))
        return done

    def reports(self):
        return [(m.body.worker_graph_ids, m.body.partition_done)
                for e, m in self.rt._communicator.sent if e == "conductor"]


def test_gated_consumer_sharing_its_worker_graph_runs_once_per_pass(tmp_path):
    """A consumer that also waits on a conductor input runs once per pass: the
    next chunk is refused until the conductor sends the next pass's input,
    which it does only after the whole worker graph reports done."""
    w = _ChunkWalkRuntime([{"node_names": ["producer", "vocoder", "post"], "ranks": [0]}], tmp_path)
    wg, = w.wgs
    assert set(wg.section.get_nodes()) == {"vocoder", "post"}
    for _ in range(3):  # three passes, one chunk each
        w.trigger()
        assert w.feed_chunk() == []
        assert w.run("vocoder") == []
        assert w.feed_chunk(), "the next chunk is refused until the next pass's trigger"
        assert w.run("post") == [wg.worker_graph_id]


def test_colocated_successor_reports_the_final_stream(tmp_path):
    """Across worker graphs on one worker, the final-stream flag is the
    worker's, so post's report, which closes the pass, carries it."""
    w = _ChunkWalkRuntime([
        {"node_names": ["producer"], "ranks": [0]},
        {"node_names": ["vocoder"], "ranks": [0]},
        {"node_names": ["post"], "ranks": [0]},
    ], tmp_path)
    voc, post = w.wg_of["vocoder"], w.wg_of["post"]
    assert voc != post
    w.trigger()
    w.feed_chunk()
    w.rt.mark_stream_partition_done(w.rid, "C")  # the vocoder step popped the final chunk
    assert w.run("vocoder") == [voc]
    assert w.run("post") == [post]
    assert w.reports() == [([voc], True), ([post], True)]


_ONE = {"ranks": [1]}
_TP2 = {"ranks": [1, 2], "tp_size": 2}


@pytest.mark.parametrize("chunk_walk,vocoder,post,accepted", [
    # post's worker never pops the final chunk, so its report closing the pass
    # lacks the flag; a self-triggered vocoder's passes also overlap post's
    (_vocoder_then_post, _ONE, {"ranks": [2]}, False),
    (_gated_vocoder_then_post, _ONE, {"ranks": [2]}, False),
    # on one worker the flag is shared, but self-triggered passes still overlap
    (_vocoder_then_post, _ONE, _ONE, False),
    (_gated_vocoder_then_post, _ONE, _ONE, True),
    # node_groups on the same ranks, replica for replica, pick one replica together
    (_gated_vocoder_then_post, {"ranks": [1, 2]}, {"ranks": [1, 2]}, True),
], ids=["self_triggered", "gated", "self_triggered_colocated", "gated_colocated", "gated_dp_replicas"])
def test_consumer_walk_across_worker_graphs(chunk_walk, vocoder, post, accepted, tmp_path, monkeypatch):
    cfg = _config(tmp_path, [
        {"node_names": ["producer"], "ranks": [0]},
        {"node_names": ["vocoder"], **vocoder},
        {"node_names": ["post"], **post},
    ])
    model = _stream_model(chunk_walk, tp_enabled_nodes=["vocoder", "post"])
    if accepted:
        c = _start_conductor(model, cfg, tmp_path, monkeypatch)
        assert c.node_walk_to_wg[("vocoder", "chunk")] is not c.node_walk_to_wg[("post", "chunk")]
        return
    with pytest.raises(ValueError, match=(
        r"graph walk 'chunk' of consumer partition 'C' consumes a stream and spans "
        r"2 worker graphs \[\['vocoder'\], \['post'\]\] on ranks"
    )):
        _start_conductor(model, cfg, tmp_path, monkeypatch)


def test_colocated_tp_consumer_walk_passes_the_stream_check_but_not_one_lockstep_node(tmp_path):
    """One TP instance on the same ranks shares the final-stream flag, but two
    TP node_groups on one worker are two lockstep nodes there, which every
    worker refused at startup and the conductor now refuses first."""
    cfg = _config(tmp_path, [
        {"node_names": ["producer"], "ranks": [0]},
        {"node_names": ["vocoder"], **_TP2},
        {"node_names": ["post"], **_TP2},
    ])
    with pytest.raises(DeploymentError) as refusal:
        _validate(_stream_model(_gated_vocoder_then_post, tp_enabled_nodes=["vocoder", "post"]), cfg)
    assert refusal.value.problems == [
        f"Multiple parallel nodes ['post', 'vocoder'] found in worker worker_{rank}; current "
        "implementation requires at most one lockstep-parallel node per worker."
        for rank in (1, 2)
    ]


def test_consumer_loop_spanning_worker_graphs_is_rejected(tmp_path):
    def looped():
        return Loop(name="chunk_loop", section=_vocoder_then_post(), max_iters=4, outputs=[])
    cfg = _config(tmp_path, [
        {"node_names": ["producer", "vocoder"], "ranks": [0]},
        {"node_names": ["post"], "ranks": [1]},
    ])
    with pytest.raises(ValueError, match="consumes a stream and spans 2 worker graphs"):
        _validate(_stream_model(looped), cfg)


def test_single_node_consumer_walks_are_accepted(tmp_path, monkeypatch):
    """The shipped shapes: a lone consumer, a consumer Loop (one top-level
    entity, as Nemotron's nano_llm loop), and a TP consumer, which is one
    worker graph on several ranks."""
    def looped():
        return Loop(name="chunk_loop", section=_vocoder(), max_iters=4, outputs=[])

    for chunk in (_vocoder, looped):
        cfg = _config(tmp_path, [{"node_names": ["producer", "vocoder"], "ranks": [0]}])
        c = _start_conductor(_stream_model(chunk), cfg, tmp_path, monkeypatch)
        assert c.node_walk_to_wg[("vocoder", "chunk")].consumes_stream

    cfg = _config(tmp_path, [
        {"node_names": ["producer"], "ranks": [0]},
        {"node_names": ["vocoder"], "ranks": [1, 2], "tp_size": 2},
    ])
    c = _start_conductor(_stream_model(_vocoder, tp_enabled_nodes=["vocoder"]), cfg, tmp_path, monkeypatch)
    assert c.node_walk_to_wg[("vocoder", "chunk")].ranks == [1, 2]


@pytest.mark.parametrize("producer_tp,vocoder_tp,shard_dim,accepted", [
    # a gathered stream: the consumer emits each chunk under a uuid its
    # StreamBuffer never registered, and no chunk is ever released
    (2, 1, {"codes": 0}, False),
    # a sliced stream: local streaming hands over the whole tensor
    (1, 2, {"codes": 0}, False),
    (2, 2, {"codes": 0}, True),
    (2, 1, {}, True),
], ids=["gathered", "sliced", "same_tp", "replicated"])
def test_sharded_stream_across_tp_sizes_is_rejected(
    producer_tp, vocoder_tp, shard_dim, accepted, tmp_path, monkeypatch,
):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({
        "max_seq_len": 64,
        "node_groups": [
            {"node_names": ["producer"], "ranks": list(range(producer_tp)), "tp_size": producer_tp},
            {"node_names": ["vocoder"], "ranks": list(range(producer_tp, producer_tp + vocoder_tp)),
             "tp_size": vocoder_tp},
        ],
        "sharding_config": {"shard_dim": shard_dim},
    }))
    model = _stream_model(_vocoder, tp_enabled_nodes=["producer", "vocoder"])
    if accepted:
        _start_conductor(model, str(path), tmp_path, monkeypatch)
        return
    with pytest.raises(RuntimeError, match=(
        rf"Streamed signal 'codes' has a shard_dim, but its producer 'producer' in graph walk 'gen' "
        rf"\(TP {producer_tp}\) and consumer 'vocoder' \(TP {vocoder_tp}\) differ in TP size"
    )):
        _start_conductor(model, str(path), tmp_path, monkeypatch)


@pytest.mark.parametrize("accumulated", [False, True], ids=["outputs", "accumulated_outputs"])
def test_a_loop_streaming_a_sharded_signal_across_tp_sizes_is_rejected(accumulated, tmp_path, monkeypatch):
    """The stream leaves through the Loop's own outputs, not the inner node's edge."""
    def generate():
        stream = StreamingGraphEdge(next_node="vocoder", name="codes", target_partition="C")
        return Loop(
            name="generate", max_iters=2,
            section=GraphNode(name="producer", input_names=["prompt"], outputs=[
                GraphEdge(next_node=EMPTY_DESTINATION, name="codes"),
            ]),
            outputs=[] if accumulated else [stream],
            accumulated_outputs=[stream] if accumulated else [],
        )

    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({
        "max_seq_len": 64,
        "node_groups": [
            {"node_names": ["producer"], "ranks": [0, 1], "tp_size": 2},
            {"node_names": ["vocoder"], "ranks": [2]},
        ],
        "sharding_config": {"shard_dim": {"codes": 0}},
    }))
    model = _stream_model(_vocoder, producer=generate, tp_enabled_nodes=["producer", "vocoder"])
    match = r"producer 'producer' in graph walk 'gen' \(TP 2\) and consumer 'vocoder' \(TP 1\)"
    with pytest.raises(RuntimeError, match=match):
        _start_conductor(model, str(path), tmp_path, monkeypatch)


def test_sharded_stream_check_uses_the_walk_that_emits_it():
    """A producer's TP group can be walk-specific: only the emitting walk's
    size counts, whichever way the other walk's size would tip it."""
    from mstar.distributed.base import ShardingGroup

    cfg = ShardingConfig(
        groups=[
            ShardingGroup(nodes={"producer"}, tp_size=2, graph_walks={"warmup"}),
            ShardingGroup(nodes={"vocoder"}, tp_size=2),
        ],
        tp_enabled_nodes={"producer", "vocoder"}, shard_dim={"codes": 0},
    )
    with pytest.raises(RuntimeError, match=r"in graph walk 'gen' \(TP 1\) and consumer 'vocoder' \(TP 2\)"):
        cfg.assert_streams_keep_tp_size([("codes", "producer", "gen", "vocoder")])
    cfg.assert_streams_keep_tp_size([("codes", "producer", "warmup", "vocoder")])


def test_conductor_driven_walk_of_a_consumer_partition_is_accepted(tmp_path, monkeypatch):
    """A consumer partition's walk with no stream-consuming node runs one pass
    per conductor input, so several worker graphs are fine there."""
    def finish():
        return Sequential([
            GraphNode(name="mix", input_names=["trigger"], outputs=[GraphEdge(next_node="post", name="wave")]),
            GraphNode(name="post", input_names=["wave"], outputs=[
                GraphEdge(next_node=EMIT_TO_CLIENT, name="audio", output_modality="audio"),
            ]),
        ])
    cfg = _config(tmp_path, [
        {"node_names": ["producer", "vocoder", "mix"], "ranks": [0]},
        {"node_names": ["post"], "ranks": [1]},
    ])
    c = _start_conductor(_stream_model(_vocoder, extra_walks={"finish": finish}), cfg, tmp_path, monkeypatch)
    assert {wg.worker_graph_id for wg in c.worker_graphs.values() if "finish" in wg.graph_walks} == {
        c.node_walk_to_wg[("mix", "finish")].worker_graph_id,
        c.node_walk_to_wg[("post", "finish")].worker_graph_id,
    }


# -- loops feeding a successor ------------------------------------------------

ITERS = 3


def _loop_feeding_post(accumulate):
    """``step`` runs ITERS times on the conductor input ``x`` and sends ``y`` to
    ``post``: from inside the Loop on every iteration, or once, through
    ``Loop.accumulated_outputs``, when the Loop finishes."""
    if accumulate:
        loop = Loop(name="loop", max_iters=ITERS, outputs=[], section=GraphNode(
            name="step", input_names={"x"}, outputs=[GraphEdge(next_node="step", name="y")]),
            accumulated_outputs=[GraphEdge(next_node="post", name="y")])
    else:
        loop = Loop(name="loop", max_iters=ITERS, outputs=[], section=GraphNode(
            name="step", input_names={"x"}, outputs=[GraphEdge(next_node="post", name="y")]))
    return Sequential([loop, GraphNode(name="post", input_names={"y"}, outputs=[
        GraphEdge(next_node=EMIT_TO_CLIENT, name="out", output_modality="text")])])


def test_loop_member_feeding_a_successor_in_its_worker_graph_is_rejected(tmp_path):
    """``post`` completes once per iteration, so the worker graph's count of
    top-level completions reaches 2 while the Loop is still running, and the
    pass-end reset drops the Loop's state and post's later inputs (#567)."""
    model = _ToyModel(lambda: {"w": _loop_feeding_post(accumulate=False)})
    cfg = _config(tmp_path, [{"node_names": ["step", "post"], "ranks": [0]}])
    with pytest.raises(ValueError, match=(
        r"graph walk 'w': node 'step' of Loop 'loop' sends 'y' to 'post', outside the Loop "
        r"in the same worker graph.*Loop.accumulated_outputs"
    )):
        _validate(model, cfg)


class _LoopWorker:
    """worker_0 hosting walk 'w' as one worker graph, driven as the worker drives it."""

    def __init__(self, runtime, tmp_path):
        model = _ToyModel(lambda: {"w": _loop_feeding_post(accumulate=True)})
        wg, = model.get_worker_graphs(_config(tmp_path, [{"node_names": ["step", "post"], "ranks": [0]}]))
        wg.worker_graph_id = 0
        args = dict(
            my_worker_id="worker_0", my_worker_graphs=[wg],
            all_wg_ids_to_graph_walks={0: {"w"}}, all_wg_ids_to_dyn_loops={0: set()},
            all_wg_ids_to_nodes={0: {"step", "post"}},
            node_to_partition={"step": "default", "post": "default"}, sharding_config=_sharding_config(),
        )
        self.runtime = runtime
        if runtime == "python":
            tm = StubTensorManager()
            # the manager's real signature: the accumulated cache holds its uuids
            tm.increment_ref = lambda uuid, n=1: tm.tensor_store.increment_ref(uuid, n=n)
            self.rt = PythonGraphRuntime(tensor_manager=tm, **args)
            self.rt._communicator = _RecordingCommunicator()
        else:
            pytest.importorskip("mstar_rust")
            from mstar.communication.tensor_store import RustTensorBookkeeping
            from mstar.graph.runtime.rust import RustGraphRuntime
            self.book = RustTensorBookkeeping()
            self.rt = RustGraphRuntime(bookkeeping=self.book, **args)
        self.rid = self.rt.add_request(
            request_id="r", partition="default", graph_walk="w", partition_worker_graph_ids=[0],
            worker_graph_to_workers=ParallelList([0], [["worker_0"]]),
            loop_caps={},
        )
        self.minter = TensorUuidMinter("worker_0")
        _ingest(self.rt, self.rid, [GraphEdge(next_node="step", name="x")])

    def _put(self, sig):
        if self.runtime == "python":
            return _store_outputs(self.rt._tensor_store, self.minter, self.rid, {s: [torch.ones(1)] for s in sig})
        uuids = {}
        for s in sig:
            u = self.minter.mint()
            self.book.put_tensor(u, TensorPointerInfo(
                dims=[1], dtype=torch.float32, nbytes=4, address=0, stride=(1,), uuid=u,
                source_session_id="s", source_entity="worker_0"))
            self.book.increment_ref(u, 1)  # the safety hold
            uuids[s] = [u]
        return uuids

    def step_once(self):
        """Run the first ready node; returns (node, reported_done) or None."""
        node = next((n for n in ("post", "step") if self.rt.pop_rids(n, "w", [self.rid], check_ready=True)), None)
        if node is None:
            return None
        rt, rid = self.rt, self.rid
        sig = rt.get_output_signals(node, "w")
        uuids = self._put(sig)
        out = rt.complete_and_route_batch(RouteInput(
            partition="default", graph_walk="w", node_name=node, output_signals=sig,
            wg_ids=ParallelList([rid], [0]), tensors=[u for s in sig for u in uuids[s]],
            num_tensors=[1] * len(sig),
        ))
        if self.runtime == "rust":
            plan = rt._rust.take_send_plan(out.completion_id)
            return node, any(r == rid and wgs for r, wgs, *_ in plan.completed)
        done = bool(rt._peek_completion(out.completion_id).routing[rid].completed_worker_graph_ids)
        rt.send_outputs(SendInput(
            completion_id=out.completion_id, new_token_counts=ParallelList([rid], [{}]),
            per_request_info=ParallelList([rid], [_fwd_info("w")]),
        ))
        return node, done


@pytest.mark.parametrize("runtime", ["python", "rust"])
def test_accumulated_outputs_feed_the_successor_once_after_the_loop(runtime, tmp_path):
    """The shape the refusal points to passes startup, and in both runtimes
    runs ``step`` ITERS times, then ``post`` once, reporting done once."""
    w = _LoopWorker(runtime, tmp_path)
    trace = []
    while len(trace) < 4 * ITERS and (r := w.step_once()):
        trace.append(r)
    assert trace == [("step", False)] * ITERS + [("post", True)], trace


# -- every shipped config -----------------------------------------------------


def _bare(cls, **attrs):
    """A model whose ``__init__`` needs the checkpoint's files: the graph
    declarations read only ``config`` (defaults here) and a few fields."""
    model = object.__new__(cls)
    model._submodule_cache = {}
    for name, value in attrs.items():
        setattr(model, name, value)
    return model


def _constructed(name, model_kwargs, **overrides):
    """A model whose ``__init__`` runs with no checkpoint (dummy mode)."""
    cls = get_model_class(name)
    kwargs = {**model_init_kwargs(name), **model_kwargs, **overrides}
    if "skip_weight_loading" in cls.__init__.__code__.co_varnames:
        kwargs["skip_weight_loading"] = True
    return cls(**kwargs)


def _bagel(name, model_kwargs, tmp_path, monkeypatch):
    from mstar.model.bagel.bagel_model import BagelModel
    from mstar.model.bagel.config import BagelAutoEncoderConfig, BagelModelConfig, BagelViTConfig
    config = BagelModelConfig(vae_config=BagelAutoEncoderConfig(), vit_config=BagelViTConfig())
    # the placement flags as __init__ leaves them; get_worker_graphs sets them
    return _bare(BagelModel, config=config, _has_cfg_parallel=False,
                 _has_llm_disaggregation=False, _image_gen_remote_handoff=False)


def _chatterbox(name, model_kwargs, tmp_path, monkeypatch):
    from mstar.model.chatterbox.chatterbox_model import ChatterboxModel
    from mstar.model.chatterbox.config import ChatterboxConfig
    config = ChatterboxConfig.from_variant(name)
    # validate_config_yaml sizes the T3 KV pool against it
    config.max_new_tokens_limit = model_kwargs.get("max_new_tokens_limit", config.max_new_tokens_limit)
    return _bare(ChatterboxModel, config=config, _t3_dtype=torch.bfloat16,
                 voices_dir=None, local_dir="/nonexistent", _shared={})


def _command_a_plus(name, model_kwargs, tmp_path, monkeypatch):
    # The tiny test config stands in for the pinned checkpoint metadata; the graph doesn't depend on its sizes.
    fixture = ROOT / "mstar/model/command_a_plus/tests/fixtures/tiny_config.json"
    (tmp_path / "config.json").write_text(fixture.read_text())
    return _constructed(name, model_kwargs, checkpoint_dir=str(tmp_path), max_seq_len=64)


def _cosmos3(name, model_kwargs, tmp_path, monkeypatch):
    if name in ("cosmos3_edge", "cosmos3_edge_droid"):
        from test_cosmos3_edge_model import _fake_edge_dir
        edge = str(_fake_edge_dir(tmp_path))
        if name == "cosmos3_edge_droid":
            # the policy fine-tune ships no reasoner vision tower
            return _constructed(name, model_kwargs, model_path_hf=edge, reasoner=None)
        return _constructed(name, model_kwargs, model_path_hf=edge)
    if name == "cosmos3_super_i2v_4step":
        # the checkpoint's transformer config has no sound pathway
        return _constructed(name, model_kwargs, sound_gen=False)
    return _constructed(name, model_kwargs)


def _fish_s2_pro(name, model_kwargs, tmp_path, monkeypatch):
    from mstar.model.fish_s2_pro.config import FishS2ProConfig
    from mstar.model.fish_s2_pro.fish_s2_pro_model import FishS2ProModel
    return _bare(FishS2ProModel, config=FishS2ProConfig())


def _flux2(name, model_kwargs, tmp_path, monkeypatch):
    from mstar.model.flux2_klein.flux2_klein_model import Flux2KleinConfig
    model = _constructed(name, model_kwargs)
    model.set_config(Flux2KleinConfig())
    return model


def _z_image(name, model_kwargs, tmp_path, monkeypatch):
    from mstar.model.z_image.z_image_model import ZImageConfig
    model = _constructed(name, model_kwargs)
    model.set_config(ZImageConfig())
    return model


def _higgs(name, model_kwargs, tmp_path, monkeypatch):
    from mstar.model.higgs_audio.higgs_audio_model import HiggsAudioModel, HiggsAudioModelConfig
    return _bare(HiggsAudioModel, config=HiggsAudioModelConfig())


def _kokoro(name, model_kwargs, tmp_path, monkeypatch):
    from mstar.model.kokoro.kokoro_model import KokoroModel, KokoroModelConfig
    return _bare(KokoroModel, config=KokoroModelConfig())


def _orpheus(name, model_kwargs, tmp_path, monkeypatch):
    from mstar.model.orpheus.orpheus_model import OrpheusModel, OrpheusModelConfig
    return _bare(OrpheusModel, config=OrpheusModelConfig())


def _ltx2_5(name, model_kwargs, tmp_path, monkeypatch):
    from mstar.model.ltx2_5.config import LTX25Config
    model = _constructed(name, model_kwargs)
    model.set_config(LTX25Config())
    return model


def _qwen3_5(name, model_kwargs, tmp_path, monkeypatch):
    """Small text-only shapes: the resource rules read the layer and head
    counts, not their sizes."""
    from mstar.model.qwen3_5.config import FULL_ATTENTION, LINEAR_ATTENTION, Qwen3_5Config
    from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel
    config = Qwen3_5Config(
        num_hidden_layers=4, hidden_size=256, intermediate_size=512,
        layer_types=[LINEAR_ATTENTION] * 3 + [FULL_ATTENTION],
        num_attention_heads=8, num_key_value_heads=2, head_dim=128,
        linear_num_key_heads=4, linear_num_value_heads=8, linear_key_head_dim=128,
        linear_value_head_dim=128, linear_conv_kernel_dim=4,
        vocab_size=1024, rms_norm_eps=1e-6, max_position_embeddings=262144,
    )
    return _bare(Qwen3_5DenseModel, config=config, vision_config=None)


def _qwen3_omni(name, model_kwargs, tmp_path, monkeypatch):
    # The talker imports flashinfer at module level; none of it runs here.
    if importlib.util.find_spec("flashinfer") is None:
        monkeypatch.setitem(sys.modules, "flashinfer", types.ModuleType("flashinfer"))
        _STUBBED_IMPORTS.append(set(sys.modules))
    from mstar.model.qwen3_omni.config import Qwen3OmniModelConfig
    from mstar.model.qwen3_omni.qwen3_omni_model import Qwen3OmniModel
    return _bare(Qwen3OmniModel, config=Qwen3OmniModelConfig())


_QWEN3_TTS_VARIANTS = {
    "qwen3_tts": ("custom_voice", "0b6"),
    "qwen3_tts_1p7b": ("custom_voice", "1b7"),
    "qwen3_tts_voicedesign": ("voice_design", "1b7"),
    "qwen3_tts_base": ("base", "1b7"),
}


def _qwen3_tts(name, model_kwargs, tmp_path, monkeypatch):
    from mstar.model.qwen3_tts.config import Qwen3TTSModelConfig
    from mstar.model.qwen3_tts.qwen3_tts_model import Qwen3TTSModel
    kind, size = _QWEN3_TTS_VARIANTS[name]
    return _bare(Qwen3TTSModel, config=Qwen3TTSModelConfig(tts_model_type=kind, tts_model_size=size))


def _qwen3_asr(name, model_kwargs, tmp_path, monkeypatch):
    from mstar.model.qwen3_asr.config import Qwen3ASRModelConfig
    from mstar.model.qwen3_asr.qwen3_asr_model import Qwen3ASRModel
    return _bare(Qwen3ASRModel, config=Qwen3ASRModelConfig())


def _whisper(name, model_kwargs, tmp_path, monkeypatch):
    from mstar.model.whisper.whisper_model import WhisperModel, WhisperModelConfig
    return _bare(WhisperModel, config=WhisperModelConfig())


def _canary(name, model_kwargs, tmp_path, monkeypatch):
    from mstar.model.canary.canary_model import CanaryModel
    from mstar.model.canary.config import CanaryModelConfig
    from mstar.model.components.fastconformer import FastConformerConfig
    config = CanaryModelConfig(encoder=FastConformerConfig())
    return _bare(CanaryModel, config=config)


def _canary_qwen(name, model_kwargs, tmp_path, monkeypatch):
    from mstar.model.canary_qwen.canary_qwen_model import CanaryQwenModel
    from mstar.model.canary_qwen.config import CanaryQwenModelConfig
    from mstar.model.components.fastconformer import FastConformerConfig
    from mstar.model.components.qwen3_lm import Qwen3LMConfig
    return _bare(CanaryQwenModel, config=CanaryQwenModelConfig(
        encoder=FastConformerConfig(), llm=Qwen3LMConfig(), llm_repo="Qwen/Qwen3-1.7B",
    ))


def _parakeet(name, model_kwargs, tmp_path, monkeypatch):
    from mstar.model.components.fastconformer import FastConformerConfig
    from mstar.model.components.transducer import TransducerConfig
    from mstar.model.parakeet.config import ParakeetModelConfig
    from mstar.model.parakeet.parakeet_model import ParakeetModel
    # Nemotron ASR conditions the transducer on a language prompt
    transducer = TransducerConfig(num_prompts=128 if name == "nemotron_asr" else 0)
    return _bare(ParakeetModel, config=ParakeetModelConfig(name, FastConformerConfig(), transducer))


def _default(name, model_kwargs, tmp_path, monkeypatch):
    return _constructed(name, model_kwargs)


_FACTORIES = {
    "BagelModel": _bagel,
    "CanaryModel": _canary,
    "CanaryQwenModel": _canary_qwen,
    "ChatterboxModel": _chatterbox,
    "CommandAPlusModel": _command_a_plus,
    "Cosmos3Model": _cosmos3,
    "FishS2ProModel": _fish_s2_pro,
    "Flux2KleinModel": _flux2,
    "ZImageModel": _z_image,
    "HiggsAudioModel": _higgs,
    "KokoroModel": _kokoro,
    "LTX25Model": _ltx2_5,
    "OrpheusModel": _orpheus,
    "ParakeetModel": _parakeet,
    "Qwen3_5DenseModel": _qwen3_5,
    "Qwen3OmniModel": _qwen3_omni,
    "Qwen3TTSModel": _qwen3_tts,
    "Qwen3ASRModel": _qwen3_asr,
    "WhisperModel": _whisper,
    # constructed directly: NemotronDuplexModel, OmniVoiceModel, Pi05Model,
    # VJepa2Model, VJepa2ACModel, Wan22Model, WaypointModel
}

_STUBBED_IMPORTS: list[set[str]] = []

SHIPPED_CONFIGS = sorted((ROOT / "configs").glob("*.yaml"))


@pytest.fixture
def _drop_stubbed_imports():
    """Modules imported against the flashinfer stub must not outlive it."""
    yield
    while _STUBBED_IMPORTS:
        before = _STUBBED_IMPORTS.pop()
        for mod in set(sys.modules) - before:
            if mod.startswith("mstar.model.qwen3_omni"):
                del sys.modules[mod]


@pytest.mark.parametrize("config_path", SHIPPED_CONFIGS, ids=lambda p: p.name)
def test_shipped_config_passes_startup_validation(config_path, tmp_path, monkeypatch, _drop_stubbed_imports):
    config = yaml.safe_load(config_path.read_text())
    name = config["model"]
    if name not in MODEL_REGISTRY:
        pytest.skip(f"model {name!r} is not in MODEL_REGISTRY")
    factory = _FACTORIES.get(MODEL_REGISTRY[name][1], _default)
    model = factory(name, config.get("model_kwargs") or {}, tmp_path, monkeypatch)
    conductor = _start_conductor(model, str(config_path), tmp_path, monkeypatch)
    assert conductor.worker_graphs
