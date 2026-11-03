"""Conductor and graph-runtime bookkeeping kept per partition and per connection
(CPU only): a colocated partition's token counts, a name persisted across
interleaved walks, and two streaming edges between one partition pair."""
import sys

sys.path.insert(0, ".")

from types import SimpleNamespace

import torch
import yaml
from test_conductor_input_merge import _config, _info
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
from mstar.conductor.request_info import CurrentForwardConductorMetadata, PartitionDefinition
from mstar.graph.base import GraphEdge, GraphNode, Loop
from mstar.graph.runtime.base import RouteInput, SendInput
from mstar.graph.runtime.python import PythonGraphRuntime
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import ForwardPassArgs, Model, WorkerGraph
from mstar.streaming.chunk_policy import FixedChunkPolicy
from mstar.streaming.topology import Connection, PartitionTopology
from mstar.utils.containers import ParallelList
from mstar.utils.ipc_format import NewRequestConductor


def colocated_partitions():
    """Partition P (a token loop) and partition C (one node) on one worker, as
    in nemotron_duplex.yaml. Shared with the Rust twin of this test."""
    graphs = [
        WorkerGraph(section=Loop(name="lp", max_iters=3, outputs=[], section=GraphNode(
            name="p", input_names=["x"], outputs=[
                GraphEdge(next_node="p", name="x"),
                GraphEdge(next_node=EMIT_TO_CLIENT, name="tok", output_modality="text",
                          conductor_new_token=True),
            ])), graph_walks={"produce"}, ranks=[0], worker_graph_id=0),
        WorkerGraph(section=GraphNode(name="c", input_names=["y"], outputs=[
            GraphEdge(next_node=EMIT_TO_CLIENT, name="audio", output_modality="audio"),
        ]), graph_walks={"consume"}, ranks=[0], worker_graph_id=1),
    ]
    return dict(
        my_worker_id="worker_0", my_worker_graphs=graphs,
        all_wg_ids_to_graph_walks={0: {"produce"}, 1: {"consume"}},
        all_wg_ids_to_dyn_loops={0: {"lp"}, 1: set()}, all_wg_ids_to_nodes={0: {"p"}, 1: {"c"}},
        node_to_partition={"p": "P", "c": "C"}, sharding_config=_sharding_config(),
    )


# P runs one loop iteration (not the last, so P's worker graph is not done) and
# emits a counted token; then C completes its one-node walk.
STEPS = [("P", "produce", 0, "p", "x"), ("C", "consume", 1, "c", "y")]


def add_both_partitions(rt):
    for part, walk, wg in [("P", "produce", 0), ("C", "consume", 1)]:
        rid = rt.add_request(
            request_id="rid", partition=part, graph_walk=walk, partition_worker_graph_ids=[wg],
            worker_graph_to_workers=ParallelList([0, 1], [["worker_0"], ["worker_0"]]),
            loop_caps={},
        )
    return rid


def test_token_counts_stay_with_their_partition():
    rt = PythonGraphRuntime(tensor_manager=StubTensorManager(), **colocated_partitions())
    rt._communicator = _RecordingCommunicator()
    rid = add_both_partitions(rt)
    minter = TensorUuidMinter("worker_0")
    for part, walk, wg, node, inp in STEPS:
        _ingest(rt, rid, [GraphEdge(next_node=node, name=inp)])
        sig = rt.get_output_signals(node, walk)
        uuids = _store_outputs(rt._tensor_store, minter, rid, {s: [torch.ones(1)] for s in sig})
        out = rt.complete_and_route_batch(RouteInput(
            partition=part, graph_walk=walk, node_name=node, output_signals=sig,
            wg_ids=ParallelList([rid], [wg]),
            tensors=[u for s in sig for u in uuids[s]], num_tensors=[1] * len(sig)))
        rt.send_outputs(SendInput(
            completion_id=out.completion_id,
            per_request_info=ParallelList([rid], [_fwd_info(walk, part)]),
            new_token_counts=ParallelList([rid], [{"tok": 1} if part == "P" else {}])))
    dones = [m.body for e, m in rt._communicator.sent if e == "conductor"]
    assert [d.partition_name for d in dones] == ["C"]
    assert dones[0].new_token_counts == {}, f"C's WORKER_GRAPHS_DONE carries P's {dones[0].new_token_counts}"
    # P's count is still held for P's own report
    assert rt._request_info[rid].pending_new_token_counts == {"P": {"tok": 1}}


def test_split_inputs_keeps_persist_order_across_walks():
    edge = GraphEdge(next_node="decoder", name="transcript", tensor_info=[
        _info(1, "prefill_prompt"), _info(2, "decode"), _info(3, "prefill_prompt")])
    got = Conductor._split_inputs_to_workers(None, _config(), [edge], "align")["worker_0"][0]
    assert [x.uuid for x in got.tensor_info] == [1, 2, 3]


class _TwoStreamModel(Model):
    def get_graph_walk_graphs(self):
        return {"p": GraphNode(name="p", input_names=["x"], outputs=[]),
                "c": GraphNode(name="c", input_names=["y"], outputs=[])}

    def get_partitions(self):
        return [PartitionDefinition("P", {"p"}), PartitionDefinition("C", {"c"})]

    def get_partition_topology(self):
        return PartitionTopology(["P", "C"], [
            Connection("P", "C", "first", lambda: FixedChunkPolicy(1)),
            Connection("P", "C", "second", lambda: FixedChunkPolicy(1)),
        ])

    def get_initial_forward_pass_args(self, partition_name, *a, **k):
        walk = "p" if partition_name == "P" else "c"
        return ForwardPassArgs(CurrentForwardConductorMetadata(graph_walk=walk, is_prefill=False), [], [])

    def get_partition_forward_pass_args(self, partition_name, partition_metadata, *a, **k):
        # P ends; C waits on its streams
        return ForwardPassArgs(partition_metadata, [], [], request_done=partition_name == "P")

    def get_node_resources(self):
        return []

    def process_prompt(self, *a, **k):
        return {}

    def postprocess(self, *a, **k):
        return b""

    def get_submodule(self, *a, **k):
        return None


class _ColonNamesModel(_TwoStreamModel):
    """Two connections whose names join to the same "P->C:x:y" string."""

    WALKS = {"P": "p", "C:x": "c1", "C": "c2"}

    def get_graph_walk_graphs(self):
        return {walk: GraphNode(name=walk, input_names=["x"], outputs=[]) for walk in self.WALKS.values()}

    def get_partitions(self):
        return [PartitionDefinition(name, {walk}) for name, walk in self.WALKS.items()]

    def get_partition_topology(self):
        return PartitionTopology(list(self.WALKS), [
            Connection("P", "C:x", "y", lambda: FixedChunkPolicy(1)),
            Connection("P", "C", "x:y", lambda: FixedChunkPolicy(1)),
        ])

    def get_initial_forward_pass_args(self, partition_name, *a, **k):
        metadata = CurrentForwardConductorMetadata(graph_walk=self.WALKS[partition_name], is_prefill=False)
        return ForwardPassArgs(metadata, [], [])


def _two_stream_conductor(tmp_path, monkeypatch, model=None):
    model = model or _TwoStreamModel()
    nodes = sorted(model.get_graph_walk_graphs())
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({"max_seq_len": 64, "node_groups": [{"node_names": nodes, "ranks": [0]}]}))
    monkeypatch.setattr(Conductor, "_launch_workers", lambda self: None)
    monkeypatch.setattr(conductor_mod, "make_communicator", lambda **kw: SimpleNamespace(send=lambda *a: None))
    c = Conductor(model, str(cfg), socket_path_prefix=str(tmp_path / "s"))
    c._ingest_request(NewRequestConductor(
        request_id="rid", initial_signals={}, initial_input_modalities=["text"],
        initial_output_modalities=["text"], input_metadata={}, model_kwargs={}))
    return c


def test_parallel_stream_connections_are_both_tracked(tmp_path, monkeypatch):
    c = _two_stream_conductor(tmp_path, monkeypatch)
    names = sorted(x.edge_name for x in c.requests["rid"].get_incoming_connections("C"))
    assert names == ["first", "second"], f"conductor tracks only {names}"


def test_producer_done_goes_once_per_consumer_partition(tmp_path, monkeypatch):
    c = _two_stream_conductor(tmp_path, monkeypatch)
    sent = []
    monkeypatch.setattr(c, "_send_producer_done", lambda rid, src, dst: sent.append((src, dst)))

    c._process_done_forward("rid", "P")

    assert sent == [("P", "C")]
    assert all(x.producer_done for x in c.requests["rid"].get_incoming_connections("C"))


def test_connections_whose_names_join_alike_are_both_tracked(tmp_path, monkeypatch):
    c = _two_stream_conductor(tmp_path, monkeypatch, _ColonNamesModel())
    sent = []
    monkeypatch.setattr(c, "_send_producer_done", lambda rid, src, dst: sent.append((src, dst)))
    req = c.requests["rid"]
    assert [x.edge_name for x in req.get_incoming_connections("C:x")] == ["y"]
    assert [x.edge_name for x in req.get_incoming_connections("C")] == ["x:y"]
    c._process_done_forward("rid", "P")
    assert sorted(sent) == [("P", "C"), ("P", "C:x")]
