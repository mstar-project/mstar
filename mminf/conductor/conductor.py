import atexit
import logging
import multiprocessing as mp
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import yaml

from mminf.api_server.types import APIServerMessage, RequestComplete
from mminf.communication.communicator import ZMQCommunicator
from mminf.graph.base import GraphPointer, TensorPointerInfo
from mminf.ipc_formats import (
    ConductorMessageType,
    InputSignals,
    NewRequest,
    NewRequestConductor,
    RemoveRequest,
    SubgraphsDone,
    UnpersistTensors,
    WorkerMessage,
    WorkerMessageType,
)
from mminf.model.base import CurrentForwardMetadata, ForwardPassArgs, Model, Subgraph

logger = logging.getLogger(__name__)


def _worker_process_target(
    worker_id: str,
    worker_ids: list[str],
    my_subgraphs: list[Subgraph],
    engine_configs: list[dict],
    all_subgraph_ids_to_phases: dict[str, set[str]],
    all_subgraph_ids_to_stages: dict[str, list[str]],
    hostname: str,
    socket_path_prefix: str,
    model: Model | None = None,
    device: str = "cuda",
    log_level: str = "INFO",
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
        sum([sg.section.get_stage_names() for sg in my_subgraphs], start=[])
    ))
    worker = Worker(
        worker_id=worker_id,
        worker_ids=worker_ids,
        my_subgraphs=my_subgraphs,
        engine_configs=engine_configs,
        all_subgraph_ids_to_phases=all_subgraph_ids_to_phases,
        all_subgraph_ids_to_stages=all_subgraph_ids_to_stages,
        hostname=hostname,
        socket_path_prefix=socket_path_prefix,
        device=torch.device(device),
        model=model,
    )
    worker.run()


@dataclass
class RequestData:
    current_forward_metadata: CurrentForwardMetadata
    fwd_inputs: list[GraphPointer]
    # name -> list[TensorPointerInfo]
    persist_signals: dict[str, list[TensorPointerInfo]] # signals passed back to conductor
    persist_signal_ref_cnt: dict[str, int] # uuid -> number of times it was passed to workers

    subgraph_to_worker: dict[str, str]
    new_tokens: dict[str, list[int]]

    # for tracking progress
    all_subgraph_ids: set[str]
    current_subgraph_ids: set[str]
    # make sure to check all tensors in the list are completed (BLOCKING case)
    completed_subgraph_ids: set[str] = field(default_factory=set)

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
        log_level: str = "INFO",
    ):
        self.requests: dict[str, RequestData] = {}
        self.model = model
        self.hostname = hostname
        self.socket_path_prefix = socket_path_prefix
        self.log_level = log_level

        self._worker_processes: list[mp.Process] = []

        with open(model_config_file, "r") as f:
            self.model_config = yaml.safe_load(f)
        assert "max_seq_len" in self.model_config
        assert "stage_groups" in self.model_config

        self.subgraphs = {
            subgraph.subgraph_id: subgraph
            for subgraph in model.get_subgraphs(model_config_file)
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
        """Derive per-rank worker info from subgraphs and model engine types."""
        stage_engine_types = self.model.get_stage_engine_types()

        # Collect unique ranks and per-rank subgraphs
        rank_to_subgraphs: dict[int, list[Subgraph]] = defaultdict(list)
        for subgraph in self.subgraphs.values():
            for rank in subgraph.ranks:
                rank_to_subgraphs[rank].append(subgraph)

        self._sorted_ranks = sorted(rank_to_subgraphs.keys())
        self.worker_ids = [f"worker_{rank}" for rank in self._sorted_ranks]

        # Per-worker subgraphs, engine configs
        self._per_worker_subgraphs: dict[str, list[Subgraph]] = {}
        self._per_worker_engine_configs: dict[str, list[dict]] = {}

        engine_model_cfg = {
            "kv_cache": self.model.get_kv_cache_config()
        }
        for rank in self._sorted_ranks:
            worker_id = f"worker_{rank}"
            subgraphs = rank_to_subgraphs[rank]
            self._per_worker_subgraphs[worker_id] = subgraphs

            # Collect engine configs: group stages by engine type
            engine_type_to_stages: dict[str, list[str]] = defaultdict(list)
            for sg in subgraphs:
                for stage_name in sg.section.get_stage_names():
                    etype = stage_engine_types[stage_name].value
                    if stage_name not in engine_type_to_stages[etype]:
                        engine_type_to_stages[etype].append(stage_name)

            self._per_worker_engine_configs[worker_id] = [
                {
                    "engine_type": etype, "stage_names": stages,
                    "model_config": engine_model_cfg
                }
                for etype, stages in engine_type_to_stages.items()
            ]

        # Global maps needed by all workers
        self._all_subgraph_ids_to_phases: dict[str, set[str]] = {
            sg_id: sg.phases for sg_id, sg in self.subgraphs.items()
        }
        self._all_subgraph_ids_to_stages: dict[str, list[str]] = {
            sg_id: sg.section.get_stage_names() for sg_id, sg in self.subgraphs.items()
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
                    "my_subgraphs": self._per_worker_subgraphs[worker_id],
                    "engine_configs": self._per_worker_engine_configs[worker_id],
                    "all_subgraph_ids_to_phases": self._all_subgraph_ids_to_phases,
                    "all_subgraph_ids_to_stages": self._all_subgraph_ids_to_stages,
                    "hostname": self.hostname,
                    "socket_path_prefix": self.socket_path_prefix,
                    "model": self.model,
                    "device": f"cuda:{rank}",
                    "log_level": self.log_level,
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

    def _assign_subgraphs_to_workers(self) -> dict[str, str]:
        """
        For a request, assign subgraphs to workers. This is relevant in the
        data parallel case, where there may be a subgraph that is replicated
        across many workers.
        """
        # Do a random policy for now. TODO: refine this
        return {
            subgraph_id: f"worker_{np.random.choice(subgraph.ranks)}"
            for subgraph_id, subgraph in self.subgraphs.items()
        }

    def _split_inputs_to_workers(
        self, subgraph_to_worker: dict[str, str],
        inputs: list[GraphPointer],
        phase: str
    ) -> dict[str, list[GraphPointer]]:
        """
        Given the full ForwardPassInputs for kicking off a new forward pass,
        return a mapping of worker_id to the ForwardPassInputs that are routed
        to that worker. ForwardPassInputs consists of graph pointers and tensors.
        """
        inputs_per_worker: dict[str, list[GraphPointer]] = {}
        for subgraph_id, worker_id in subgraph_to_worker.items():
            subgraph = self.subgraphs[subgraph_id]
            if phase not in subgraph.phases:
                continue
            stages = set(subgraph.section.get_stage_names())

            if worker_id not in inputs_per_worker:
                inputs_per_worker[worker_id] = []
            inputs_per_worker[worker_id] += [
                ptr for ptr in inputs if ptr.next_stage in stages
            ]
        return inputs_per_worker

    def _update_request_info(
        self, request_id, args: ForwardPassArgs
    ):
        self._set_current_subgraph_ids(
            request_id,
            args.full_metadata.phase
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
        subgraph (for all possible execution phases, e.g., prefill, decode, image_gen),
        and notify the workers that the request has arrived + provide the appropriate
        workers with the appropriate initial inputs for the forward pass
        """
        logger.debug("Conductor ingesting request %s", body.request_id)
        subgraph_to_worker = self._assign_subgraphs_to_workers()
        request_data = RequestData(
            current_forward_metadata=None,
            fwd_inputs=[],
            persist_signals=body.initial_signals,
            persist_signal_ref_cnt={},
            subgraph_to_worker=subgraph_to_worker,
            all_subgraph_ids=set(subgraph_to_worker.keys()),
            current_subgraph_ids=set(),
            new_tokens={},
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
        worker_to_subgraph_ids: dict[str, list[str]] = {}
        inputs_per_worker = self._split_inputs_to_workers(
            subgraph_to_worker=subgraph_to_worker,
            inputs=fwd_args.inputs,
            phase=fwd_args.full_metadata.phase
        )

        for subgraph_id, worker_id in subgraph_to_worker.items():
            if worker_id not in worker_to_subgraph_ids:
                worker_to_subgraph_ids[worker_id] = []
            worker_to_subgraph_ids[worker_id].append(subgraph_id)

        for worker, subgraph_ids in worker_to_subgraph_ids.items():
            message = NewRequest(
                request_id=body.request_id,
                subgraph_ids=subgraph_ids,
                subgraph_to_worker=subgraph_to_worker,
                initial_phase=request_data.current_forward_metadata.phase,
                initial_inputs=inputs_per_worker.get(worker, []),
                per_request_metadata=fwd_args.step_metadata,
            )
            self.communicator.send(
                worker, WorkerMessage(
                    message_type=WorkerMessageType.NEW_REQUEST,
                    body=message
                )
            )

    def _set_current_subgraph_ids(
        self, request_id: str, phase: str
    ):
        self.requests[request_id].current_subgraph_ids = set([
            sg for sg in self.requests[request_id].all_subgraph_ids \
                if phase in self.subgraphs[sg].phases
        ])

    def _process_request_done(
        self, request_id: str
    ):
        """
        Called when we see an EOS token, e.g.
        """
        logger.info("Request %s done", request_id)
        for worker_id in set(self.requests[request_id].subgraph_to_worker.values()):
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
                    request_id=request_id
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

        prev_phase = request_data.current_forward_metadata.phase
        fwd_args = self.model.get_forward_pass_args(
            request_data.current_forward_metadata,
            persist_signals=request_data.persist_signals,
            new_tokens=request_data.new_tokens
        )
        self._update_request_info(request_id, fwd_args)

        logger.debug(
            ("Request %s completed forward pass; moving from phase %s -> %s.\n"
             "Received new tokens %s, has persist signals %s.\n"
             "request_done=%s"),
            request_id, prev_phase, fwd_args.full_metadata.phase,
            str(request_data.new_tokens), str(list(request_data.persist_signals.keys())),
            str(fwd_args.request_done)
        )
        self._un_persist_tensors(request_id, fwd_args.unpersist_tensors)
        if fwd_args.request_done:
            return True # stop the request

        logger.debug("Forward inputs: %s", str(fwd_args.inputs))

        inputs_per_worker = self._split_inputs_to_workers(
            subgraph_to_worker=request_data.subgraph_to_worker,
            inputs=fwd_args.inputs,
            phase=fwd_args.full_metadata.phase
        )

        for worker, inputs in inputs_per_worker.items():
            message = WorkerMessage(
                message_type=WorkerMessageType.INPUT_SIGNALS,
                body=InputSignals(
                    request_id=request_id,
                    phase=request_data.current_forward_metadata.phase,
                    inputs=inputs,
                    per_request_metadata=fwd_args.step_metadata,
                )
            )
            self.communicator.send(worker, message)

        request_data.new_tokens = {}
        request_data.completed_subgraph_ids = set()
        return False

    def _process_subgraphs_done(
        self, body: SubgraphsDone
    ):
        """
        When some subgraphs have completed (the worker notifies the conductor that
        the subgraphs have completed), update the metadata for this request.

        Return whether the full model forward pass has been completed (i.e., all
        subgraphs for the current computation phase have been completed)
        """
        request_data = self.requests[body.request_id]

        # Absorb persist signals and new tokens sent with this message
        if body.persist_signals:
            request_data.persist_signals.update(body.persist_signals)
        if body.new_tokens:
            for name in body.new_tokens:
                if name not in request_data.new_tokens:
                    request_data.new_tokens[name] = []
                request_data.new_tokens[name] += body.new_tokens[name]

        request_data.completed_subgraph_ids.update(
            body.subgraph_ids
        )

        done_with_forward = request_data.current_subgraph_ids.issubset(
            request_data.completed_subgraph_ids
        )
        return done_with_forward

    def run(self):
        while True:
            try:
                done_forward_passes = []
                for message in self.communicator.get_all_new_messages():
                    if message.message_type == ConductorMessageType.NEW_REQUEST:
                        self._ingest_request(message.body)
                    elif message.message_type == ConductorMessageType.SUBGRAPHS_DONE:
                        done_with_fwd = self._process_subgraphs_done(
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

            time.sleep(0.001)
