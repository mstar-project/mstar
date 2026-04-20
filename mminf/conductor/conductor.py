import atexit
import hashlib
import logging
import multiprocessing as mp
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch
import yaml

from mminf.api_server.request_types import APIServerMessage, RequestComplete
from mminf.communication.communicator import CommProtocol, ZMQCommunicator
from mminf.conductor.request_info import (
    CurrentForwardConductorMetadata,
    CurrentForwardPassInfo,
    PartitionDefinition,
    PartitionState,
    StreamingConnectionState,
)
from mminf.engine.base import EngineType
from mminf.engine.kv_store import KVCacheConfig
from mminf.graph.base import GraphEdge, TensorPointerInfo
from mminf.model.base import ForwardPassArgs, Model, WorkerGraph
from mminf.utils.ipc_format import (
    ConductorMessageType,
    InputSignals,
    NewRequest,
    NewRequestConductor,
    RemoveRequest,
    UnpersistTensors,
    WorkerGraphsDone,
    WorkerMessage,
    WorkerMessageType,
)
from mminf.utils.sampling import SamplingConfig

logger = logging.getLogger(__name__)


def _req_id_to_seed(req_id: str):
    """Map a request id to a 32-bit seed.

    Uses ``hashlib.md5`` rather than Python's builtin ``hash`` so the result
    is **stable across processes**: Python salts ``hash`` per-interpreter via
    ``PYTHONHASHSEED``, which would otherwise make the conductor's per-request
    seed unpredictable from a client process. A deterministic mapping lets a
    client pin ``request_id`` and reproduce the exact noise the server's
    sampler will use, which is essential for noise-controlled debugging
    (e.g. comparing Pi0.5 server output against a reference implementation).
    """
    digest = hashlib.md5(req_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def _worker_process_target(
    worker_id: str,
    worker_ids: list[str],
    my_worker_graphs: list[WorkerGraph],
    kv_config: list[KVCacheConfig],
    model_config: dict,
    all_worker_graph_ids_to_graph_walks: dict[str, set[str]],
    all_worker_graph_ids_to_nodes: dict[str, set[str]],
    all_worker_graph_ids_to_dyn_loops: dict[str, set[str]],
    hostname: str,
    socket_path_prefix: str,
    enable_nvtx: bool = False,
    model: Model | None = None,
    device: str = "cuda",
    log_level: str = "INFO",
    mooncake_port: int=8080,
    tensor_comm_protocol=CommProtocol.RDMA,
    tcp_transfer_device="",
):
    """Top-level target for spawned worker processes. Must be module-level for picklability."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=f"%(asctime)s %(levelname)s [{worker_id}] %(name)s: %(message)s",
        force=True,
    )

    from mminf.worker.worker import Worker
    logger.debug("Launching worker %s with graph nodes %s", worker_id, str(
        [wg.section.get_node_names() for wg in my_worker_graphs]
    ))
    worker = Worker(
        worker_id=worker_id,
        worker_ids=worker_ids,
        model=model,
        my_worker_graphs=my_worker_graphs,
        kv_config=kv_config,
        model_config=model_config,
        all_worker_graph_ids_to_graph_walks=all_worker_graph_ids_to_graph_walks,
        all_worker_graph_ids_to_nodes=all_worker_graph_ids_to_nodes,
        all_worker_graph_ids_to_dyn_loops=all_worker_graph_ids_to_dyn_loops,
        hostname=hostname,
        socket_path_prefix=socket_path_prefix,
        enable_nvtx=enable_nvtx,
        device=torch.device(device),
        mooncake_port=mooncake_port,
        tensor_comm_protocol=tensor_comm_protocol,
        tcp_transfer_device=tcp_transfer_device,
    )
    worker.run()


@dataclass
class RequestData:
    # Request-level shared state
    persist_signals: dict[str, list[TensorPointerInfo]]  # signals passed back to conductor
    persist_signal_ref_cnt: dict[str, int]  # uuid -> number of times it was passed to workers
    worker_graph_to_worker: dict[str, str]
    all_worker_graph_ids: set[str]
    max_output_tokens: int
    random_seed: int
    sampling_config: dict[str, SamplingConfig | None]

    # Partition state (always populated — single-partition models use a "default" partition)
    partition_states: dict[str, PartitionState] = field(default_factory=dict)
    partition_definitions: dict[str, PartitionDefinition] = field(default_factory=dict)

    # Per-streaming-connection state (keyed by "from_partition->to_partition")
    streaming_connections: dict[str, StreamingConnectionState] = field(default_factory=dict)

    def remove_persist_signal_uuids(self, uuids: list[str]):
        uuids = set(uuids)
        for name in self.persist_signals:
            self.persist_signals[name] = [
                info for info in self.persist_signals[name] if info.uuid not in uuids
            ]

        for uuid in uuids:
            del self.persist_signal_ref_cnt[uuid]

    def get_incoming_connections(self, partition_name: str) -> list[StreamingConnectionState]:
        """Return all streaming connections where the given partition is the consumer."""
        return [
            conn for conn in self.streaming_connections.values()
            if conn.to_partition == partition_name
        ]


class Conductor:
    """
    Initial in-progress conductor implementation. TODO: this is extremely
    un-optimized, but it provides a sense of the data movement between the
    conductor and the workers
    """
    def __init__(
        self,
        model: Model,
        model_config_file: str,
        socket_path_prefix: str = "/tmp/mminf",
        hostname: str = "localhost",
        enable_nvtx: bool = False,
        log_level: str = "INFO",
        mooncake_port: int=8080,
        tensor_comm_protocol=CommProtocol.RDMA,
        tcp_transfer_device=""
    ):
        self.requests: dict[str, RequestData] = {}
        self.model = model
        self.hostname = hostname
        self.socket_path_prefix = socket_path_prefix
        self.log_level = log_level
        self.enable_nvtx = enable_nvtx
        self.mooncake_port = mooncake_port
        self.tensor_comm_protocol = tensor_comm_protocol
        self.tcp_transfer_device = tcp_transfer_device

        self._worker_processes: list[mp.Process] = []
        self.waiting_queue: list[NewRequestConductor] = []

        with open(model_config_file, "r") as f:
            self.model_config = yaml.safe_load(f)
        self.max_concurrent_requests: int = self.model_config.get(
            "max_concurrent_requests", None
        )
        assert "max_seq_len" in self.model_config
        assert "node_groups" in self.model_config

        self.worker_graphs = {
            worker_graph.worker_graph_id: worker_graph
            for worker_graph in model.get_worker_graphs(model_config_file)
        }

        os.makedirs(socket_path_prefix, exist_ok=True)
        self._derive_worker_info()
        self._launch_workers()

        self.communicator = ZMQCommunicator(
            my_id="conductor",
            push_ids=self.worker_ids + ["api_server", "api_server_preprocess_worker"],
            ipc_socket_path_prefix=socket_path_prefix,
        )

    def _get_kv_config(self):
        kv_cache_config = self.model.get_kv_cache_config()
        # Apply any KV cache overrides from the YAML config
        yaml_kv_overrides = self.model_config.get("kv_cache", {})
        if yaml_kv_overrides:
            from dataclasses import fields
            for kv_cfg in kv_cache_config:
                for f in fields(kv_cfg):
                    if f.name in yaml_kv_overrides:
                        setattr(kv_cfg, f.name, yaml_kv_overrides[f.name])
                logger.info("KV cache config after YAML overrides: %s", kv_cfg)
        return kv_cache_config
    
    def _get_sampling_configs(self, model_kwargs: dict):
        ar_nodes = [
            node for (node, engine) in self.model.get_node_engine_types().items() \
                if engine in {EngineType.AR, EngineType.CODE_PREDICTOR}
        ]
        return {
            node: self.model.get_sampling_config(
                node_name=node, model_kwargs=model_kwargs
            ) for node in ar_nodes
        }

    def _derive_worker_info(self):
        """Derive per-rank worker info from worker graphs and model engine types."""
        node_engine_types = self.model.get_node_engine_types()

        # Collect unique ranks and per-rank worker graphs
        rank_to_worker_graphs: dict[int, list[WorkerGraph]] = defaultdict(list)
        for worker_graph in self.worker_graphs.values():
            for rank in worker_graph.ranks:
                rank_to_worker_graphs[rank].append(worker_graph)

        self._sorted_ranks = sorted(rank_to_worker_graphs.keys())
        self.worker_ids = [f"worker_{rank}" for rank in self._sorted_ranks]

        # Per-worker graph units, engine configs
        self._per_worker_graphs: dict[str, list[WorkerGraph]] = {}

        for rank in self._sorted_ranks:
            worker_id = f"worker_{rank}"
            worker_graphs = rank_to_worker_graphs[rank]
            self._per_worker_graphs[worker_id] = worker_graphs

        # Global maps needed by all workers
        self._all_worker_graph_ids_to_graph_walks: dict[str, set[str]] = {
            worker_graph_id: worker_graph.graph_walks for worker_graph_id, worker_graph in self.worker_graphs.items()
        }
        self._all_worker_graph_ids_to_nodes: dict[str, set[str]] = {
            worker_graph_id: worker_graph.section.get_node_names()
            for worker_graph_id, worker_graph in self.worker_graphs.items()
        }
        self._all_worker_graph_ids_to_dyn_loops = {
            worker_graph_id: worker_graph.section.get_dyn_loop_names()
            for worker_graph_id, worker_graph in self.worker_graphs.items()
        }

    def _launch_workers(self):
        """Spawn one process per worker rank using spawn context."""
        ctx = mp.get_context("spawn")
        for rank, worker_id in zip(self._sorted_ranks, self.worker_ids, strict=True):
            p = ctx.Process(
                target=_worker_process_target,
                kwargs={
                    "worker_id": worker_id,
                    "worker_ids": self.worker_ids,
                    "my_worker_graphs": self._per_worker_graphs[worker_id],
                    "kv_config": self._get_kv_config(),
                    "model_config": self.model_config,
                    "all_worker_graph_ids_to_graph_walks": self._all_worker_graph_ids_to_graph_walks,
                    "all_worker_graph_ids_to_nodes": self._all_worker_graph_ids_to_nodes,
                    "all_worker_graph_ids_to_dyn_loops": self._all_worker_graph_ids_to_dyn_loops,
                    "hostname": self.hostname,
                    "socket_path_prefix": self.socket_path_prefix,
                    "model": self.model,
                    "enable_nvtx": self.enable_nvtx,
                    "device": f"cuda:{rank}",
                    "log_level": self.log_level,
                    "mooncake_port": self.mooncake_port,
                    "tensor_comm_protocol": self.tensor_comm_protocol,
                    "tcp_transfer_device": self.tcp_transfer_device
                },
                daemon=False,
            )
            p.start()
            self._worker_processes.append(p)

        atexit.register(self.shutdown)

    def shutdown(self):
        logger.info("Shutting down conductor...")
        """Terminate and join all worker processes."""
        for p in self._worker_processes:
            if p.is_alive():
                p.terminate()
        for p in self._worker_processes:
            p.join(timeout=5)
        self._worker_processes.clear()

    def _assign_worker_graphs_to_workers(self) -> dict[str, str]:
        """
        For a request, assign worker graphs to workers. This is relevant in the
        data parallel case, where there may be a worker graph that is replicated
        across many workers.
        """
        # Do a random policy for now. TODO: refine this
        return {
            worker_graph_id: f"worker_{np.random.choice(worker_graph.ranks)}"
            for worker_graph_id, worker_graph in self.worker_graphs.items()
        }

    def _split_inputs_to_workers(
        self, worker_graph_to_worker: dict[str, str],
        inputs: list[GraphEdge],
        graph_walk: str
    ) -> dict[str, list[GraphEdge]]:
        """
        Given the full ForwardPassInputs for kicking off a new forward pass,
        return a mapping of worker_id to the ForwardPassInputs that are routed
        to that worker. ForwardPassInputs consists of graph edges and tensors.
        """
        inputs_per_worker: dict[str, list[GraphEdge]] = {}
        for worker_graph_id, worker_id in worker_graph_to_worker.items():
            worker_graph = self.worker_graphs[worker_graph_id]
            if graph_walk not in worker_graph.graph_walks:
                continue
            nodes = set(worker_graph.section.get_node_names())

            if worker_id not in inputs_per_worker:
                inputs_per_worker[worker_id] = []
            inputs_per_worker[worker_id] += [
                edge for edge in inputs if edge.next_node in nodes
            ]
        return inputs_per_worker

    def _update_persist_ref_counts(
        self, request_id: str, inputs: list[GraphEdge]
    ):
        """Update reference counts for persist signals in inputs."""
        ref_cnts = self.requests[request_id].persist_signal_ref_cnt
        for edge in inputs:
            for info in edge.tensor_info:
                if info.uuid not in ref_cnts:
                    ref_cnts[info.uuid] = 0
                ref_cnts[info.uuid] += 1

    def _un_persist_tensors(
        self, request_id: str, tensor_info: list[TensorPointerInfo]
    ):
        entity_id_to_msg = {}
        uuids = []
        for info in tensor_info:
            uuid_to_ref_count = entity_id_to_msg.setdefault(
                info.source_entity, UnpersistTensors(
                    request_id=request_id, uuid_to_ref_count={}
                )
            ).uuid_to_ref_count

            if info.uuid in uuid_to_ref_count:
                # duplicate; skip
                continue
            ref_cnt = self.requests[request_id].persist_signal_ref_cnt.get(info.uuid)
            if ref_cnt is None:
                continue  # tensor not tracked (e.g., from a different partition)
            uuid_to_ref_count[info.uuid] = ref_cnt
            uuids.append(info.uuid)
        self.requests[request_id].remove_persist_signal_uuids(uuids)

        for (entity, body) in entity_id_to_msg.items():
            self.communicator.send(
                entity, WorkerMessage(
                    message_type=WorkerMessageType.UNPERSIST_TENSORS,
                    body=body
                )
            )

    def _try_admit_waiting(self):
        """Drain the waiting queue up to the concurrency cap."""
        while self.waiting_queue:
            if (self.max_concurrent_requests is not None
                    and len(self.requests) >= self.max_concurrent_requests):
                break
            body = self.waiting_queue.pop(0)
            logger.info(
                "Admitting queued request %s (%d/%s in-flight)",
                body.request_id, len(self.requests),
                str(self.max_concurrent_requests),
            )
            self._do_ingest_request(body)

    def _ingest_request(
        self, body: NewRequestConductor
    ):
        """
        When a new request comes in from the API server, assign workers,
        initialize partition states, and kick off all partitions.
        """
        if (self.max_concurrent_requests is not None
                and len(self.requests) >= self.max_concurrent_requests):
            logger.info(
                "Request %s queued (at capacity: %d/%d)",
                body.request_id, len(self.requests),
                self.max_concurrent_requests,
            )
            self.waiting_queue.append(body)
            return
        self._do_ingest_request(body)

    def _do_ingest_request(
        self, body: NewRequestConductor
    ):
        """Actually dispatch a request to workers (no admission check)."""
        logger.debug("Conductor ingesting request %s", body.request_id)
        worker_graph_to_worker = self._assign_worker_graphs_to_workers()

        model_kwargs = body.model_kwargs or {}
        max_output_tokens = self.model.get_max_output_tokens(**model_kwargs)
        seed = _req_id_to_seed(body.request_id)

        partitions = self.model.get_partitions()
        topology = self.model.get_partition_topology()

        # Build partition states and definitions
        partition_states: dict[str, PartitionState] = {}
        partition_definitions: dict[str, PartitionDefinition] = {}
        for p in partitions:
            partition_definitions[p.name] = p
            partition_states[p.name] = PartitionState(
                partition_name=p.name,
                metadata=CurrentForwardConductorMetadata(
                    input_modalities=body.initial_input_modalities,
                    output_modalities=body.initial_output_modalities,
                    graph_walk="",
                    is_prefill=True,
                ),
                random_seed=seed,
            )

        # Build per-connection streaming state
        streaming_connections: dict[str, StreamingConnectionState] = {}
        for conn in topology.connections:
            key = f"{conn.from_partition}->{conn.to_partition}"
            streaming_connections[key] = StreamingConnectionState(
                from_partition=conn.from_partition,
                to_partition=conn.to_partition,
                edge_name=conn.edge_name,
            )

        request_data = RequestData(
            persist_signals=body.initial_signals,
            persist_signal_ref_cnt={},
            worker_graph_to_worker=worker_graph_to_worker,
            all_worker_graph_ids=set(worker_graph_to_worker.keys()),
            max_output_tokens=max_output_tokens,
            random_seed=seed,
            partition_states=partition_states,
            partition_definitions=partition_definitions,
            streaming_connections=streaming_connections,
            sampling_config=self._get_sampling_configs(model_kwargs)
        )
        self.requests[body.request_id] = request_data

        # Collect all worker_graph_ids per worker for the NewRequest
        worker_to_worker_graph_ids: dict[str, list[str]] = defaultdict(list)
        for wg_id, worker_id in worker_graph_to_worker.items():
            worker_to_worker_graph_ids[worker_id].append(wg_id)

        # Kick off all partitions by calling get_initial_forward_pass_args per partition
        partition_fwd_args: dict[str, ForwardPassArgs] = {}
        for p in partitions:
            fwd_args = self.model.get_initial_forward_pass_args(
                partition_name=p.name,
                input_modalities=body.initial_input_modalities,
                output_modalities=body.initial_output_modalities,
                input_signals=body.initial_signals,
                model_kwargs=body.model_kwargs,
            )
            pstate = partition_states[p.name]
            # if a partition is not active at all in the request, register that here
            pstate.is_done = fwd_args.request_done

            pstate.metadata = fwd_args.full_metadata
            pstate.metadata.kwargs.update(fwd_args.step_metadata)
            self._set_partition_worker_graph_ids(
                body.request_id, p.name, fwd_args.full_metadata.graph_walk,
            )
            self._update_persist_ref_counts(body.request_id, fwd_args.inputs)
            partition_fwd_args[p.name] = fwd_args

        # Send NewRequest to each worker with the appropriate partition's inputs
        for worker_id, worker_graph_ids in worker_to_worker_graph_ids.items():
            # Determine which partition this worker serves
            for partition_name, partition_wg_ids in self._resolve_worker_partition(
                worker_graph_ids, partitions,
            ).items():
                fwd_args = partition_fwd_args[partition_name]
                pstate = partition_states[partition_name]
                inputs_per_worker = self._split_inputs_to_workers(
                    worker_graph_to_worker=worker_graph_to_worker,
                    inputs=fwd_args.inputs,
                    graph_walk=fwd_args.full_metadata.graph_walk,
                )

                message = NewRequest(
                    request_id=body.request_id,
                    partition_worker_graph_ids=partition_wg_ids,
                    worker_graph_to_worker=worker_graph_to_worker,
                    initial_inputs=inputs_per_worker.get(worker_id, []),
                    request_info=CurrentForwardPassInfo(
                        graph_walk=fwd_args.full_metadata.graph_walk,
                        step_metadata=fwd_args.step_metadata,
                        fwd_index=pstate.fwd_pass_number,
                        random_seed=pstate.random_seed,
                        requires_cfg=fwd_args.full_metadata.requires_cfg,
                        partition_name=partition_name,
                        max_tokens=request_data.max_output_tokens,
                        sampling_config=request_data.sampling_config
                    ),
                )
                self.communicator.send(
                    worker_id, WorkerMessage(
                        message_type=WorkerMessageType.NEW_REQUEST,
                        body=message,
                    ),
                )

    def _resolve_worker_partition(
        self, worker_graph_ids: list[str],
        partitions: list[PartitionDefinition],
    ) -> dict[str, set[str]]:
        """Find which partition(s) a set of worker graphs belongs to."""
        partition_wg_ids = {}
        for wg_id in worker_graph_ids:
            wg_walks = self._all_worker_graph_ids_to_graph_walks.get(wg_id, set())
            for p in partitions:
                if wg_walks & p.graph_walks:
                    partition_wg_ids.setdefault(p.name, set()).add(wg_id)
        return partition_wg_ids

    def _set_partition_worker_graph_ids(
        self, request_id: str, partition_name: str, graph_walk: str,
    ):
        """Update the set of active worker graph IDs for a partition's walk."""
        pstate = self.requests[request_id].partition_states[partition_name]
        pstate.current_worker_graph_ids = {
            wg_id for wg_id in self.requests[request_id].all_worker_graph_ids
            if graph_walk in self.worker_graphs[wg_id].graph_walks
        }

    def _process_request_done(
        self, request_id: str
    ):
        """Called when all partitions are done."""
        logger.info("Request %s done", request_id)
        request_data = self.requests[request_id]
        for worker_id in set(request_data.worker_graph_to_worker.values()):
            msg = WorkerMessage(
                message_type=WorkerMessageType.REMOVE_REQUEST,
                body=RemoveRequest(request_id)
            )
            self.communicator.send(worker_id, msg)

        # Build output dict: output_name -> final forward pass number
        final_forward_outputs: dict[str, int] = {}
        for pstate in request_data.partition_states.values():
            for output_name in pstate.curr_forward_outputs:
                # fwd_pass_number was already incremented past the last pass
                final_forward_outputs[output_name] = max(0, pstate.fwd_pass_number - 1)

        self.communicator.send(
            "api_server",
            APIServerMessage(
                message_type="request_complete",
                body=RequestComplete(
                    request_id=request_id,
                    final_forward_outputs=final_forward_outputs,
                )
            )
        )
        del self.requests[request_id]
        self._try_admit_waiting()

    def _process_worker_graphs_done(
        self, body: WorkerGraphsDone
    ) -> list[str]:
        """Process a WorkerGraphsDone message.

        Uses the partition_name from the message directly.
        Returns list of partition names whose full forward pass has completed.
        """
        if body.request_id not in self.requests:
            logger.debug(
                "Ignoring late WORKER_GRAPHS_DONE for completed request %s",
                body.request_id
            )
            return []

        request_data = self.requests[body.request_id]
        partition_name = body.partition_name

        pstate = request_data.partition_states.get(partition_name)
        if pstate is None:
            logger.warning(
                "WorkerGraphsDone for unknown partition %s (request %s)",
                partition_name, body.request_id,
            )
            return []

        # Update sequence info
        pstate.per_label_seq_info.update(body.per_label_seq_info)

        # Absorb persist signals (request-level)
        if body.persist_signals:
            request_data.persist_signals.update(body.persist_signals)

        # Absorb new tokens and update streaming connection state
        if body.new_tokens:
            for name, tokens in body.new_tokens.items():
                pstate.new_tokens.setdefault(name, []).extend(tokens)
                pstate.num_output_tokens += len(tokens)
                # Update streaming connections where this partition is producer
                for conn in request_data.streaming_connections.values():
                    if conn.from_partition == partition_name and conn.edge_name == name:
                        conn.token_count += len(tokens)

        # Update consumed counts from worker-reported stream consumption
        if body.stream_tokens_consumed:
            for conn in request_data.streaming_connections.values():
                if conn.from_partition == partition_name:
                    continue  # skip producer connections
                consumed = body.stream_tokens_consumed.get(conn.edge_name, 0)
                conn.consumed_count = max(conn.consumed_count, consumed)

        pstate.completed_worker_graph_ids.update(body.worker_graph_ids)
        pstate.curr_forward_outputs += body.output_signal_names if isinstance(
            body.output_signal_names, list
        ) else []

        # Check if this partition's forward pass is fully done
        done_partitions = []
        if pstate.current_worker_graph_ids.issubset(pstate.completed_worker_graph_ids):
            done_partitions.append(partition_name)

        return done_partitions

    def _process_done_forward(
        self, request_id: str, partition_name: str,
        partition_done_from_worker: bool = False,
    ) -> bool:
        """Process a completed forward pass for a specific partition.

        Calls get_partition_forward_pass_args for all partitions uniformly.
        If the result has inputs, sends them. If not, the partition
        self-triggers (e.g., via StreamBuffer on the worker).

        Returns True if the **entire** request is done (all partitions finished).
        """
        request_data = self.requests[request_id]
        pstate = request_data.partition_states[partition_name]

        incoming_connections = request_data.get_incoming_connections(partition_name)

        # For partitions that self-trigger via StreamBuffer (have incoming
        # connections with topology), worker signals partition_done directly.
        if incoming_connections and partition_done_from_worker:
            pstate.is_done = True

        prev_walk =  pstate.metadata.graph_walk
        fwd_args = self.model.get_partition_forward_pass_args(
            partition_name=partition_name,
            partition_metadata=pstate.metadata,
            persist_signals=request_data.persist_signals,
            new_tokens=pstate.new_tokens,
            incoming_connections=incoming_connections,
        )
        pstate.metadata = fwd_args.full_metadata
        pstate.metadata.kwargs.update(fwd_args.step_metadata)

        # Check max output tokens for partitions that produce tokens
        if pstate.num_output_tokens >= request_data.max_output_tokens:
            logger.info(
                "Partition %s reached max output tokens %d. Ending.",
                partition_name, request_data.max_output_tokens,
            )
            fwd_args.request_done = True

        logger.debug(
            "Partition %s of request %s: %s -> %s (request_done=%s, tokens=%d)",
            partition_name, request_id, prev_walk,
            fwd_args.full_metadata.graph_walk, fwd_args.request_done,
            pstate.num_output_tokens,
        )

        if fwd_args.request_done:
            pstate.is_done = True
            # Signal producer_done to all outgoing connections
            for conn in request_data.streaming_connections.values():
                if conn.from_partition == partition_name:
                    conn.producer_done = True
                    self._send_producer_done(request_id, conn.from_partition, conn.to_partition)
        elif fwd_args.inputs:
            # Partition has inputs to send — conductor-driven
            self._send_partition_inputs(request_id, partition_name, fwd_args)
        # else: no inputs — partition self-triggers via StreamBuffer

        self._un_persist_tensors(request_id, fwd_args.unpersist_tensors)

        # Reset partition forward pass state
        pstate.new_tokens = {}
        pstate.completed_worker_graph_ids = set()
        pstate.current_worker_graph_ids = set()
        pstate.fwd_pass_number += 1
        pstate.random_seed += 1

        self._set_partition_worker_graph_ids(
            request_id, partition_name, fwd_args.full_metadata.graph_walk,
        )

        # Request done when ALL partitions are done
        return all(ps.is_done for ps in request_data.partition_states.values())

    def _send_partition_inputs(
        self, request_id: str, partition_name: str, fwd_args: ForwardPassArgs,
    ):
        """Send InputSignals for a specific partition's next forward pass."""
        request_data = self.requests[request_id]
        pstate = request_data.partition_states[partition_name]

        self._update_persist_ref_counts(request_id, fwd_args.inputs)

        inputs_per_worker = self._split_inputs_to_workers(
            worker_graph_to_worker=request_data.worker_graph_to_worker,
            inputs=fwd_args.inputs,
            graph_walk=fwd_args.full_metadata.graph_walk,
        )

        for worker, inputs in inputs_per_worker.items():
            message = WorkerMessage(
                message_type=WorkerMessageType.INPUT_SIGNALS,
                body=InputSignals(
                    request_id=request_id,
                    inputs=inputs,
                    request_info=CurrentForwardPassInfo(
                        graph_walk=fwd_args.full_metadata.graph_walk,
                        step_metadata=fwd_args.step_metadata,
                        fwd_index=pstate.fwd_pass_number,
                        random_seed=pstate.random_seed,
                        per_label_seq_info=pstate.per_label_seq_info,
                        requires_cfg=fwd_args.full_metadata.requires_cfg,
                        partition_name=partition_name,
                        max_tokens=request_data.max_output_tokens,
                        sampling_config=request_data.sampling_config,
                    ),
                    partition_name=partition_name
                ),
            )
            self.communicator.send(worker, message)

    def _send_producer_done(
        self, request_id: str, producer_partition: str,
        consumer_partition_name: str
    ):
        """Send producer_done signal to the consumer partition's worker(s)."""
        request_data = self.requests[request_id]
        pstate = request_data.partition_states[consumer_partition_name]

        # Find which workers handle this consumer partition
        consumer_workers = set()
        pdef = request_data.partition_definitions[consumer_partition_name]
        for wg_id, worker_id in request_data.worker_graph_to_worker.items():
            walks = self._all_worker_graph_ids_to_graph_walks.get(wg_id, set())
            if walks & pdef.graph_walks:
                consumer_workers.add(worker_id)

        for worker_id in consumer_workers:
            message = WorkerMessage(
                message_type=WorkerMessageType.INPUT_SIGNALS,
                body=InputSignals(
                    request_id=request_id,
                    inputs=[],
                    request_info=CurrentForwardPassInfo(
                        graph_walk=pstate.metadata.graph_walk or "",
                        fwd_index=pstate.fwd_pass_number,
                        random_seed=pstate.random_seed,
                        requires_cfg=False,
                        partition_name=consumer_partition_name,
                        max_tokens=request_data.max_output_tokens,
                        sampling_config=request_data.sampling_config
                    ),
                    partition_name=consumer_partition_name,
                    producer_done=set([producer_partition]),
                ),
            )
            self.communicator.send(worker_id, message)

    def run(self):
        from mminf.utils.profiler import range_pop, range_push

        while True:
            if self.enable_nvtx:
                range_push("conductor.run_loop")

            try:
                done_partition_forwards: list[tuple[str, str, bool]] = []

                for message in self.communicator.get_all_new_messages():
                    if message.message_type == ConductorMessageType.NEW_REQUEST:
                        self._ingest_request(message.body)
                    elif message.message_type == ConductorMessageType.WORKER_GRAPHS_DONE:
                        rid = message.body.request_id
                        if rid not in self.requests:
                            logger.debug(
                                "WORKER_GRAPHS_DONE for unknown request %s (already completed?)", rid
                            )
                            continue

                        done_parts = self._process_worker_graphs_done(message.body)
                        for pname in done_parts:
                            done_partition_forwards.append(
                                (rid, pname, message.body.partition_done)
                            )
                    else:
                        raise ValueError(f"Unknown message type: {message.message_type}")

                completed_requests = []

                for request_id, partition_name, p_done in done_partition_forwards:
                    if request_id not in self.requests:
                        continue  # already completed by another partition in this cycle
                    all_done = self._process_done_forward(
                        request_id, partition_name,
                        partition_done_from_worker=p_done,
                    )
                    if all_done:
                        completed_requests.append(request_id)

                for request_id in dict.fromkeys(completed_requests):
                    if request_id in self.requests:
                        self._process_request_done(request_id)
            except Exception:
                logger.exception("Conductor error in main loop")
            finally:
                if self.enable_nvtx:
                    range_pop()

            time.sleep(0.001)
