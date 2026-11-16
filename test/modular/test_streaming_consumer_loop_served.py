"""A loop that consumes a stream runs once per chunk when it is served, the stream's end included.

CosyVoice3's token2wav is such a loop: the flow takes a chunk of the LM's speech tokens and HiFT vocodes what the flow
made, each looping its own state back. The worker releases a step's consumed inputs before it routes the step's
outputs, so the flow's loop-backs found its slot free and went into the iteration that had run, as did a chunk that
came before HiFT ended it, and the loop dropped both when it advanced. The flow never ran again, and no request longer
than one chunk finished.

These serve a model's own worker graphs on one worker, as its shipped config places them, on the Rust runtime the
worker builds, with the worker's intake and stream buffers and the model's chunk policy between producer and consumer.
Cosmos3's decoder loops over its window stream the same way.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch
import yaml
from rust_runtime import build_runtime
from test_graph_runtime import _sharding_config
from test_startup_graph_validation import _FACTORIES

from mstar.communication.tensors import SharedMemoryCommunicationManager
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.graph.base import GraphEdge
from mstar.graph.runtime.base import RouteInput
from mstar.testing_utils.worker import make_worker
from mstar.utils.containers import ParallelList
from mstar.utils.ipc_format import InputSignals, NewRequest
from mstar.worker.node_manager_utils import RequestStateManager
from mstar.worker.stream_buffers import StreamBufferManager

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
WORKER = "worker_0"
RID = "r"


class _Case(NamedTuple):
    model: str
    config: str
    producer: str
    consumer: str
    items: int
    # how many items each step of the stream's consumer node reads
    reads: list[int]


CASES = {
    # a chunk every 25 tokens (stream_poll_tokens), carrying every token before it
    "cosyvoice3": _Case("CosyVoice3Model", "cosyvoice3.yaml", "decode", "token2wav", 60, [25, 50, 60]),
    "cosmos3": _Case("Cosmos3Model", "cosmos3_nano_ar.yaml", "video_gen_ar", "video_decode_ar", 3, [1, 1, 1]),
}


class _StubEngines:
    """Holds no resources, so taking a request in opens nothing."""

    engine = SimpleNamespace(publisher_nodes={})

    def add_request(self, *args) -> None:
        pass


class _Served:
    """A model on one worker, its nodes faked: each writes a tensor on every output. The loop of ``case.producer``
    streams ``case.items`` items into the loop of ``case.consumer``."""

    def __init__(self, tmp_path, case: _Case):
        deployment = yaml.safe_load((CONFIGS / case.config).read_text())
        model = _FACTORIES[case.model](deployment["model"], deployment.get("model_kwargs") or {}, tmp_path, None)
        graphs = model.get_worker_graphs(str(CONFIGS / case.config))
        sections = {walk: wg.section for wg in graphs for walk in wg.graph_walks}
        self.wg = {walk: wg.worker_graph_id for wg in graphs for walk in wg.graph_walks}
        self.partition = {walk: p.name for p in model.get_partitions() for walk in p.graph_walks}
        node_to_partition = {
            node: self.partition[walk] for wg in graphs for walk in wg.graph_walks for node in wg.section.get_nodes()
        }
        self.tm = SharedMemoryCommunicationManager(
            my_entity_id=WORKER, hostname="localhost", device="cpu",
            communicator=SimpleNamespace(send=lambda *a, **k: None), shm_dir=str(tmp_path),
        )
        self.rt = build_runtime(
            bookkeeping=self.tm.tensor_store.bookkeeping, my_worker_id=WORKER, my_worker_graphs=graphs,
            all_wg_ids_to_graph_walks={wg.worker_graph_id: wg.graph_walks for wg in graphs},
            all_wg_ids_to_dyn_loops={wg.worker_graph_id: set(wg.section.get_loops()) for wg in graphs},
            all_wg_ids_to_nodes={wg.worker_graph_id: set(wg.section.get_nodes()) for wg in graphs},
            node_to_partition=node_to_partition, sharding_config=_sharding_config(),
        )
        topology = model.get_partition_topology()
        (connection,) = topology.connections
        self.stream = connection.edge_name
        request_state = RequestStateManager(node_to_partition=node_to_partition)
        streams = StreamBufferManager(
            request_state, self.tm, topology.connections, model.get_graph_walk_graphs(),
            graph_runtime=self.rt, topology=topology,
        )
        self.worker = make_worker(
            worker_id=WORKER, graph_runtime=self.rt, tensor_manager=self.tm, request_state=request_state,
            streams=streams, engine_manager=_StubEngines(),
            wakeup_event=SimpleNamespace(register_futures=lambda futures: None),
        )
        for walk in (case.producer, case.consumer):
            # what the conductor opens a walk with: its inputs, and an empty edge for each loop-back
            io = sections[walk].get_inputs_outputs()
            entry = sorted((name, node) for name, node in io.ext_inputs | io.loop_back if name != self.stream)
            self.worker.ingest._add_new_request(NewRequest(
                request_id=RID, partition_worker_graph_ids=[self.wg[walk]],
                worker_graph_to_workers={wg.worker_graph_id: [WORKER] for wg in graphs},
                initial_inputs=[GraphEdge(name=name, next_node=node) for name, node in entry],
                request_info=CurrentForwardPassInfo(
                    request_id=RID, graph_walk=walk, fwd_index=0, random_seed=0, max_tokens=case.items,
                    partition_name=self.partition[walk],
                ),
                loop_caps={},
            ))
        self.rid = self.rt.get_rid_handle(RID)
        self.walk_of = {node: walk for walk in (case.producer, case.consumer) for node in sections[walk].get_nodes()}
        self.writes = {
            node.name: list(dict.fromkeys(edge.name for edge in node.outputs))
            for walk in (case.producer, case.consumer) for node in sections[walk].get_nodes().values()
        }
        ((self.producer_loop, producer_loop),) = sections[case.producer].get_loops().items()
        ((self.consumer_loop, consumer_loop),) = sections[case.consumer].get_loops().items()
        (self.producer,) = producer_loop.get_nodes()
        self.consumers = list(consumer_loop.get_nodes())
        self.reader = streams.consumer_node[self.stream]
        self.consumer_partition = self.partition[case.consumer]
        self.case = case
        self.reads: list[int] = []
        self._made = 0

    def serve(self, order: tuple[str, ...]) -> None:
        """Serves the request until the consumer's loop ends, a step a pass: each pass polls the stream buffers, as the
        worker's preamble does, then runs the first node of ``order`` that is ready."""
        for _ in range(4 * self.case.items):
            if self.rt.is_partition_idle(self.rid, self.consumer_partition):
                return
            self.worker.ingest.poll_stream_buffers()
            ready = {spec.node_name for spec in self.rt.get_ready_nodes(set())}
            node = next((node for node in order if node in ready), None)
            assert node is not None, f"nothing is ready and the request is open, after {self.reader} steps {self.reads}"
            if node == self.producer:
                self._produce()
            else:
                self._step(node)
        raise AssertionError(f"the consumer's loop never ended, after {self.reader} steps {self.reads}")

    def _produce(self) -> None:
        self._made += 1
        last = self._made == self.case.items
        self._step(self.producer, stop=self.producer_loop if last else None)
        if last:
            # the conductor ends the stream once the producer's partition ends, counting what it streamed here
            self.worker.ingest._process_new_inputs(InputSignals(
                request_id=RID, inputs=[], request_info=None, partition_name=self.consumer_partition,
                producer_done={self.partition[self.case.producer]}, producer_done_items={self.stream: self.case.items},
            ))

    def _step(self, node: str, stop: str | None = None) -> None:
        """One step of ``node`` in the order `StepExecutor.postprocess` keeps: its consumed inputs are released, its
        loop stop registered, then its outputs routed, the producer's items into the consumer's stream buffer."""
        walk = self.walk_of[node]
        edges = self.rt.pop_rids(node, walk, [self.rid]).input_edges
        if node == self.reader:
            k = edges.signal_name_idxs.index(edges.signal_names.index(self.stream))
            self.reads.append(edges.tensors_per_edge[k])
            if edges.is_final_streaming_chunk[k]:
                # the worker ends a consumer's loop on the step that takes its stream's last chunk
                stop = self.consumer_loop
        self.tm.cleanup_collectable(*self.rt.cleanup_consumed_inputs(node, [self.rid], [self.wg[walk]]))
        if stop is not None:
            self.rt.stop_loops_batched(
                partition=self.partition[walk], graph_walk=walk, last_node_run=node,
                loop_names=ParallelList([self.rid], [[stop]]),
            )
        signals = self.writes[node]
        stored = self.tm.store_and_return_tensor_info_batch(
            [self.rid], {self.rid: {signal: [torch.ones(1)] for signal in signals}}, signals,
        )
        self.tm.increment_ref_batch_uniform(stored.flat_uuids, 1)
        out = self.rt.complete_and_route_batch(RouteInput(
            partition=self.partition[walk], graph_walk=walk, node_name=node, output_signals=signals,
            wg_ids=ParallelList([self.rid], [self.wg[walk]]), tensors=stored.flat_uuids,
            num_tensors=stored.num_tensors,
        ))
        self.tm.cleanup_collectable(*out.freed_inputs)
        buffers = self.worker.request_state.per_request_info[self.rid].stream_buffers
        for signal, per_signal in out.local_streaming_by_signal.items():
            uuids = [uuid for _rid, uuid in per_signal]
            buffers[signal].pre_read_register_joined(uuids)
            for uuid in uuids:
                buffers[signal].put(uuid, uuid)


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_a_stream_longer_than_a_chunk_runs_the_consumer_loop_once_per_chunk(case, tmp_path):
    served = _Served(tmp_path, case)
    # the consumer's nodes whenever they are ready and the producer otherwise, as CosyVoice3's GPU run went
    served.serve(order=(*served.consumers, served.producer))
    assert served.reads == case.reads, "the consumer should run once per chunk, the stream's end included"


def test_a_chunk_that_comes_before_the_loop_advances_waits_for_the_next_iteration(tmp_path):
    case = CASES["cosyvoice3"]
    served = _Served(tmp_path, case)
    # the LM streams every token, and the conductor ends the stream, between the flow's first step and HiFT's
    served.serve(order=(served.reader, served.producer, *served.consumers))
    assert served.reads == case.reads, "a chunk taken into the iteration that ran is dropped with it"
