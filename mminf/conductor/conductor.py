import atexit
import logging
import multiprocessing as mp
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import yaml

from mminf.api_server.request_types import APIServerMessage, RequestComplete
from mminf.communication.communicator import ZMQCommunicator
from mminf.engine.base import EngineType
from mminf.engine.kv_store import SequenceInfo
from mminf.graph.base import GraphEdge, TensorPointerInfo
from mminf.model.base import DECODE, CurrentForwardMetadata, ForwardPassArgs, Model, WorkerGraph
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

logger = logging.getLogger(__name__)


def _worker_process_target(
    worker_id: str,
    worker_ids: list[str],
    my_worker_graphs: list[WorkerGraph],
    engine_configs: list[dict],
    all_worker_graph_ids_to_graph_walks: dict[str, set[str]],
    all_worker_graph_ids_to_nodes: dict[str, list[str]],
    hostname: str,
    socket_path_prefix: str,
    enable_nvtx: bool = False,
    model: Model | None = None,
    device: str = "cuda",
    log_level: str = "INFO",
    mooncake_port: int=8080
):
    """Top-level target for spawned worker processes. Must be module-level for picklability."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=f"%(asctime)s %(levelname)s [{worker_id}] %(name)s: %(message)s",
        force=True,
    )
    import torch

    from mminf.worker.worker import Worker
    logger.debug("Launching worker %s with graph nodes %s", worker_id, str(
        sum([wg.section.get_node_names() for wg in my_worker_graphs], start=[])
    ))
    worker = Worker(
        worker_id=worker_id,
        worker_ids=worker_ids,
        my_worker_graphs=my_worker_graphs,
        engine_configs=engine_configs,
        all_worker_graph_ids_to_graph_walks=all_worker_graph_ids_to_graph_walks,
        all_worker_graph_ids_to_nodes=all_worker_graph_ids_to_nodes,
        hostname=hostname,
        socket_path_prefix=socket_path_prefix,
        enable_nvtx=enable_nvtx,
        device=torch.device(device),
        model=model,
        mooncake_port=mooncake_port
    )
    worker.run()


@dataclass
class RequestData:
    current_forward_metadata: CurrentForwardMetadata
    fwd_inputs: list[GraphEdge]
    # name -> list[TensorPointerInfo]
    persist_signals: dict[str, list[TensorPointerInfo]] # signals passed back to conductor
    persist_signal_ref_cnt: dict[str, int] # uuid -> number of times it was passed to workers

    worker_graph_to_worker: dict[str, str]
    new_tokens: dict[str, list[int]]

    # for tracking progress
    all_worker_graph_ids: set[str]
    current_worker_graph_ids: set[str]
    max_output_tokens: int
    num_output_tokens: int = field(default=0)
    # make sure to check all tensors in the list are completed (BLOCKING case)
    completed_worker_graph_ids: set[str] = field(default_factory=set)
    fwd_pass_number: int = field(default=0)
    curr_forward_outputs: list[str] = field(default_factory=list)
    per_label_seq_info: dict[str, SequenceInfo] = field(default_factory=dict)

    def remove_persist_signal_uuids(self, uuids: list[str]):
        uuids = set(uuids)
        for name in self.persist_signals:
            self.persist_signals[name] = [
                info for info in self.persist_signals[name] if info.uuid not in uuids
            ]

        for uuid in uuids:
            del self.persist_signal_ref_cnt[uuid]


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
        mooncake_port: int=8080
    ):
        self.requests: dict[str, RequestData] = {}
        self.model = model
        self.hostname = hostname
        self.socket_path_prefix = socket_path_prefix
        self.log_level = log_level
        self.enable_nvtx = enable_nvtx
        self.mooncake_port = mooncake_port

        self._worker_processes: list[mp.Process] = []

        with open(model_config_file, "r") as f:
            self.model_config = yaml.safe_load(f)
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
        self._per_worker_engine_configs: dict[str, list[dict]] = {}

        engine_model_cfg = {
            "kv_cache": self.model.get_kv_cache_config(),
            "autocast_dtype": self.model.get_autocast_dtype()
        }
        for rank in self._sorted_ranks:
            worker_id = f"worker_{rank}"
            worker_graphs = rank_to_worker_graphs[rank]
            self._per_worker_graphs[worker_id] = worker_graphs

            # Collect engine configs: group nodes by engine type
            engine_type_to_nodes: dict[str, list[str]] = defaultdict(list)
            for wg in worker_graphs:
                for node_name in wg.section.get_node_names():
                    etype = node_engine_types[node_name].value
                    if node_name not in engine_type_to_nodes[etype]:
                        engine_type_to_nodes[etype].append(node_name)

            self._per_worker_engine_configs[worker_id] = [
                {
                    "engine_type": etype, "node_names": nodes,
                    "model_config": engine_model_cfg
                }
                for etype, nodes in engine_type_to_nodes.items()
            ]

        # Global maps needed by all workers
        self._all_worker_graph_ids_to_graph_walks: dict[str, set[str]] = {
            worker_graph_id: worker_graph.graph_walks for worker_graph_id, worker_graph in self.worker_graphs.items()
        }
        self._all_worker_graph_ids_to_nodes: dict[str, list[str]] = {
            worker_graph_id: worker_graph.section.get_node_names()
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
                    "engine_configs": self._per_worker_engine_configs[worker_id],
                    "all_worker_graph_ids_to_graph_walks": self._all_worker_graph_ids_to_graph_walks,
                    "all_worker_graph_ids_to_nodes": self._all_worker_graph_ids_to_nodes,
                    "hostname": self.hostname,
                    "socket_path_prefix": self.socket_path_prefix,
                    "model": self.model,
                    "enable_nvtx": self.enable_nvtx,
                    "device": f"cuda:{rank}",
                    "log_level": self.log_level,
                    "mooncake_port": self.mooncake_port
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

    def _update_request_info(
        self, request_id, args: ForwardPassArgs
    ):
        self._set_current_worker_graph_ids(
            request_id,
            args.full_metadata.graph_walk
        )
        self.requests[request_id].current_forward_metadata = args.full_metadata
        self.requests[request_id].fwd_inputs = args.inputs

        # update reference counts for persist signals
        ref_cnts = self.requests[request_id].persist_signal_ref_cnt
        for edge in args.inputs:
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
            uuid_to_ref_count[info.uuid] = self.requests[
                request_id].persist_signal_ref_cnt[info.uuid]
            uuids.append(info.uuid)
        self.requests[request_id].remove_persist_signal_uuids(uuids)

        for (entity, body) in entity_id_to_msg.items():
            self.communicator.send(
                entity, WorkerMessage(
                    message_type=WorkerMessageType.UNPERSIST_TENSORS,
                    body=body
                )
            )

    def _ingest_request(
        self, body: NewRequestConductor
    ):
        """
        When a new request comes in from the API server, assign workers for each
        worker graph for all possible graph walks, e.g., prefill, decode, image_gen),
        and notify the workers that the request has arrived + provide the appropriate
        workers with the appropriate initial inputs for the forward pass.
        """
        logger.debug("Conductor ingesting request %s", body.request_id)
        worker_graph_to_worker = self._assign_worker_graphs_to_workers()

        # set up request data
        model_kwargs = body.model_kwargs or {}
        max_output_tokens = self.model.get_max_output_tokens(**model_kwargs)
        request_data = RequestData(
            current_forward_metadata=None,
            fwd_inputs=[],
            persist_signals=body.initial_signals,
            persist_signal_ref_cnt={},
            worker_graph_to_worker=worker_graph_to_worker,
            all_worker_graph_ids=set(worker_graph_to_worker.keys()),
            current_worker_graph_ids=set(),
            new_tokens={},
            max_output_tokens=max_output_tokens,
        )
        self.requests[body.request_id] = request_data

        fwd_args = self.model.get_initial_forward_pass_args(
            input_modalities=body.initial_input_modalities,
            output_modalities=body.initial_output_modalities,
            input_signals=body.initial_signals,
            model_kwargs=body.model_kwargs
        )
        self._update_request_info(body.request_id, fwd_args)

        # send data to appropriate workers
        worker_to_worker_graph_ids: dict[str, list[str]] = {}
        inputs_per_worker = self._split_inputs_to_workers(
            worker_graph_to_worker=worker_graph_to_worker,
            inputs=fwd_args.inputs,
            graph_walk=fwd_args.full_metadata.graph_walk
        )

        for worker_graph_id, worker_id in worker_graph_to_worker.items():
            if worker_id not in worker_to_worker_graph_ids:
                worker_to_worker_graph_ids[worker_id] = []
            worker_to_worker_graph_ids[worker_id].append(worker_graph_id)

        for worker, worker_graph_ids in worker_to_worker_graph_ids.items():
            message = NewRequest(
                request_id=body.request_id,
                worker_graph_ids=worker_graph_ids,
                worker_graph_to_worker=worker_graph_to_worker,
                initial_graph_walk=request_data.current_forward_metadata.graph_walk,
                initial_inputs=inputs_per_worker.get(worker, []),
                per_request_metadata=fwd_args.step_metadata,
            )
            self.communicator.send(
                worker, WorkerMessage(
                    message_type=WorkerMessageType.NEW_REQUEST,
                    body=message
                )
            )

    def _set_current_worker_graph_ids(
        self, request_id: str, graph_walk: str
    ):
        self.requests[request_id].current_worker_graph_ids = set([
            worker_graph_id for worker_graph_id in self.requests[request_id].all_worker_graph_ids \
                if graph_walk in self.worker_graphs[worker_graph_id].graph_walks
        ])

    def _process_request_done(
        self, request_id: str
    ):
        """
        Called when we see an EOS token, e.g.
        """
        logger.info("Request %s done", request_id)
        for worker_id in set(self.requests[request_id].worker_graph_to_worker.values()):
            msg = WorkerMessage(
                message_type=WorkerMessageType.REMOVE_REQUEST,
                body=RemoveRequest(request_id)
            )
            self.communicator.send(worker_id, msg)
        self.communicator.send(
            "api_server",
            APIServerMessage(
                message_type="request_complete",
                body=RequestComplete(
                    request_id=request_id,
                    final_forward_pass=self.requests[request_id].fwd_pass_number,
                    final_forward_outputs=self.requests[request_id].curr_forward_outputs
                )
            )
        )
        del self.requests[request_id]

    def _process_done_forward(
        self, request_id: str
    ) -> bool:
        """
        If the request isn't over, start a new forward pass (determine the input and
        output modalities for the new forward pass, wrangle input tensors and send
        them to the appropriate workers)

        Returns a boolean for whether the request is done
        """
        request_data = self.requests[request_id]

        prev_graph_walk = request_data.current_forward_metadata.graph_walk
        fwd_args = self.model.get_forward_pass_args(
            request_data.current_forward_metadata,
            persist_signals=request_data.persist_signals,
            new_tokens=request_data.new_tokens
        )
        self._update_request_info(request_id, fwd_args)

        logger.debug(
            ("Request %s completed forward pass; moving from graph_walk %s -> %s.\n"
             "Received new tokens %s, has persist signals %s.\n"
             "request_done=%s"),
            request_id, prev_graph_walk, fwd_args.full_metadata.graph_walk,
            str(request_data.new_tokens), str(list(request_data.persist_signals.keys())),
            str(fwd_args.request_done)
        )
        self._un_persist_tensors(request_id, fwd_args.unpersist_tensors)

        if request_data.num_output_tokens >= request_data.max_output_tokens:
            logger.info(
                "Request %s reached max output tokens %d. Ending request.",
                request_id, request_data.max_output_tokens
            )
            fwd_args.request_done = True
        if fwd_args.request_done:
            return True # stop the request
        
        request_data.fwd_pass_number += 1
        request_data.curr_forward_outputs.clear()

        logger.debug("Forward inputs: %s", str(fwd_args.inputs))

        inputs_per_worker = self._split_inputs_to_workers(
            worker_graph_to_worker=request_data.worker_graph_to_worker,
            inputs=fwd_args.inputs,
            graph_walk=fwd_args.full_metadata.graph_walk
        )

        for worker, inputs in inputs_per_worker.items():
            message = WorkerMessage(
                message_type=WorkerMessageType.INPUT_SIGNALS,
                body=InputSignals(
                    request_id=request_id,
                    graph_walk=request_data.current_forward_metadata.graph_walk,
                    inputs=inputs,
                    per_request_metadata=fwd_args.step_metadata,
                    fwd_pass_number=request_data.fwd_pass_number,
                    per_label_seq_info=self.requests[request_id].per_label_seq_info
                )
            )
            self.communicator.send(worker, message)

        request_data.new_tokens = {}
        request_data.completed_worker_graph_ids = set()
        return False

    def _process_worker_graphs_done(
        self, body: WorkerGraphsDone
    ):
        """
        When some worker graphs have completed (the worker notifies the conductor
        that the worker graphs have completed), update the metadata for this
        request.

        Return whether the full model forward pass has been completed (i.e., all
        worker graphs for the current computation graph walk have been completed)
        """
        request_data = self.requests[body.request_id]

        request_data.per_label_seq_info = {
            **request_data.per_label_seq_info,
            **body.per_label_seq_info
        }

        # Absorb persist signals and new tokens sent with this message
        if body.persist_signals:
            request_data.persist_signals.update(body.persist_signals)
        if body.new_tokens:
            for name in body.new_tokens:
                if name not in request_data.new_tokens:
                    request_data.new_tokens[name] = []
                request_data.new_tokens[name] += body.new_tokens[name]
                request_data.num_output_tokens += len(body.new_tokens[name])

        request_data.completed_worker_graph_ids.update(
            body.worker_graph_ids
        )
        request_data.curr_forward_outputs += body.output_signal_names

        done_with_forward = request_data.current_worker_graph_ids.issubset(
            request_data.completed_worker_graph_ids
        )
        return done_with_forward

    def run(self):
        from mminf.utils.profiler import range_pop, range_push

        while True:
            if self.enable_nvtx:
                range_push("conductor.run_loop")

            try:
                done_forward_passes = []
                for message in self.communicator.get_all_new_messages():
                    if message.message_type == ConductorMessageType.NEW_REQUEST:
                        self._ingest_request(message.body)
                    elif message.message_type == ConductorMessageType.WORKER_GRAPHS_DONE:
                        done_with_fwd = self._process_worker_graphs_done(
                            message.body
                        )
                        if done_with_fwd:
                            done_forward_passes.append(message.body.request_id)
                    else:
                        raise ValueError(f"Unknown message type: {message.message_type}")

                completed_requests = []
                for request_id in done_forward_passes:
                    saw_eos = self._process_done_forward(request_id)
                    if saw_eos:
                        completed_requests.append(request_id)

                for request_id in completed_requests:
                    self._process_request_done(request_id)
            except Exception:
                logger.exception("Conductor error in main loop")
            finally:
                if self.enable_nvtx:
                    range_pop()

            time.sleep(0.001)
