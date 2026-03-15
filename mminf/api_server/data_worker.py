

import logging
import queue
import threading
import time
from dataclasses import asdict

import torch
import torchaudio
import torchvision

try:
    from torchcodec.decoders import VideoDecoder
except (ImportError, RuntimeError):
    VideoDecoder = None

from mminf.api_server.request_types import PreprocessInput, ResultChunk, ResultTensors
from mminf.communication.communicator import CommProtocol, ZMQCommunicator
from mminf.communication.tensors import MooncakeCommunicationManager, NameToTensorList
from mminf.model.base import Model
from mminf.utils.ipc_format import (
    ConductorMessage,
    ConductorMessageType,
    NewRequestConductor,
    TensorReceived,
    UnpersistTensors,
    WorkerMessageType,
)

logger = logging.getLogger(__name__)


def _preprocess_loop(**kwargs):
    worker = PreprocessWorkerThread(**kwargs)
    worker.run()


class PreprocessWorker:
    def __init__(
        self,
        model: Model | None = None,
        hostname: str = "localhost",
        socket_path_prefix: str = "/tmp/mminf",
        tensor_comm_protocol: CommProtocol = CommProtocol.RDMA,
    ):
        self.request_input_queue = queue.Queue()
        self.result_tensor_input_queue = queue.Queue()
        self.cleanup_request_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.stop_event = threading.Event()

        self.per_request_reading_tensors = {}

        self.thread = threading.Thread(
            target=_preprocess_loop,
            kwargs=dict(
                in_queue=self.request_input_queue,
                result_tensor_queue=self.result_tensor_input_queue,
                out_queue=self.output_queue,
                cleanup_request_queue=self.cleanup_request_queue,
                stop_event=self.stop_event,
                hostname=hostname,
                socket_path_prefix=socket_path_prefix,
                tensor_comm_protocol=tensor_comm_protocol,
                model=model,
            )
        )
        self.thread.start()

    def new_request(self, input: PreprocessInput):
        self.per_request_reading_tensors[input.request_id] = 0
        self.request_input_queue.put(input)

    def new_result_tensors(self, input: ResultTensors):
        self.per_request_reading_tensors[input.request_id] += len(input.graph_edge.tensor_info)
        logger.debug(
            "Data worker reading queue for request %s increased to length %d",
            input.request_id,  self.per_request_reading_tensors[input.request_id]
        )
        self.result_tensor_input_queue.put(input)

    def has_pending_tensors(self, request_id: str):
        return self.per_request_reading_tensors.get(request_id, 0) > 0

    def get_result_chunks(self)-> list[ResultChunk]:
        results = []
        while not self.output_queue.empty():
            result: ResultChunk = self.output_queue.get()
            self.per_request_reading_tensors[result.request_id] -= 1
            logger.debug(
                "Data worker reading queue for request %s decreased to length %d",
                result.request_id,  self.per_request_reading_tensors[result.request_id]
            )
            results.append(result)
        return results

    def cleanup_request(self, request_id: str):
        self.cleanup_request_queue.put(request_id)

    def shutdown(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join()


class PreprocessWorkerThread:
    def __init__(
        self,
        in_queue: queue.Queue, # for preprocessing
        result_tensor_queue: queue.Queue, # for output streaming
        out_queue: queue.Queue,
        cleanup_request_queue: queue.Queue,
        stop_event: threading.Event,
        hostname: str = "localhost",
        socket_path_prefix: str = "/tmp/mminf",
        tensor_comm_protocol: CommProtocol = CommProtocol.RDMA,
        device: str = "cpu",
        model: Model | None = None,
    ):
        self.in_queue = in_queue
        self.result_tensor_queue = result_tensor_queue
        self.cleanup_request_queue = cleanup_request_queue
        self.out_queue = out_queue

        self.stop_event = stop_event
        self.device = device
        self.model = model

        self.tensor_uuid_to_metadata_per_request = {}

        self.communicator = ZMQCommunicator(
            my_id="api_server_preprocess_worker",
            push_ids=["conductor"],
            ipc_socket_path_prefix=socket_path_prefix,
        ) # only used to send

        self.tensor_manager = MooncakeCommunicationManager(
            my_entity_id="api_server_preprocess_worker",
            hostname=hostname,
            communicator=self.communicator,
            protocol=tensor_comm_protocol,
        )

    def _process_input(
        self, input: PreprocessInput
    ):
        tensors: NameToTensorList = {}
        input_metadata = {}

        # Tokenize prompt via model (model-specific tokenization + system prompt)
        if self.model is not None:
            prompt_tensors = self.model.process_prompt(
                input.text,
                input.input_modalities,
                input.output_modalities,
                **(input.model_kwargs or {}),
            )
            tensors.update(prompt_tensors)
        elif input.text is not None:
            # Fallback: encode as UTF-8 bytes -> uint8 tensor
            byte_data = input.text.encode("utf-8")
            tensors["text_inputs"] = [torch.tensor(
                list(byte_data), dtype=torch.uint8, device=self.device
            )]

        if input.file_paths is not None:
            for modality in input.file_paths:
                key = f"{modality}_inputs"
                tensors[key] = []
                # TODO: maybe make a class of tensors_and_metadata later (figure out how to use metadata)
                input_metadata[key] = []

                for filepath in input.file_paths[modality]:
                    # ---- Image ----
                    if modality == "image":
                        img = torchvision.io.decode_image(filepath).to(self.device)  # uint8 CxHxW
                        img = img.float() / 255.0
                        tensors[key].append(img)
                        input_metadata[key].append({}) # cleanest, no metadata

                    # ---- Audio ----
                    elif modality == "audio":
                        waveform, sample_rate = torchaudio.load_with_torchcodec(
                            filepath,
                            channels_first=True
                        )
                        # waveform: (channels, time)
                        tensors[key].append(waveform)
                        input_metadata[key].append(dict(
                            sample_rate=sample_rate,
                            channels_first=True
                        ))

                    # ---- Video ----
                    elif modality == "video":
                        decoder = VideoDecoder(filepath, device=self.device)
                        video = torch.stack([frame for frame in decoder]).float() / 255.0
                        tensors[key].append(video)
                        input_metadata[key].append(asdict(decoder.metadata))

        initial_signals = self.tensor_manager.store_and_return_tensor_info(
            request_id=input.request_id,
            tensors=tensors # dict(modality_input: list[tensors])
        )
        all_uuids = sum([
            [info.uuid for info in infos] for infos in initial_signals.values()
        ], start=[])
        self.tensor_manager.register_for_send(
            request_id=input.request_id,
            uuids=all_uuids
        )
        # also persist all of the input signals
        for uuid in all_uuids:
            self.tensor_manager.set_persist(
                input.request_id, uuid, persist=True
            )

        msg = ConductorMessage(
            message_type=ConductorMessageType.NEW_REQUEST,
            body=NewRequestConductor(
                request_id=input.request_id,
                initial_signals=initial_signals,
                initial_input_modalities=input.input_modalities,
                initial_output_modalities=input.output_modalities,
                input_metadata=input_metadata,
                model_kwargs=input.model_kwargs
            ),
        )
        self.communicator.send("conductor", msg)

    def _read_result_tensor(
        self, result: ResultTensors
    ):
        result.graph_edge.name = f"{result.modality}_output"
        self.tensor_manager.start_read_tensors(
            request_id=result.request_id,
            graph_edges=[result.graph_edge],
            device=self.device
        )
        if result.request_id not in self.tensor_uuid_to_metadata_per_request:
            self.tensor_uuid_to_metadata_per_request[result.request_id] = {}
        for tensor_info in result.graph_edge.tensor_info:
            self.tensor_uuid_to_metadata_per_request[result.request_id][
                tensor_info.uuid] = result.metadata

    def _process_read_tensors(self):
        for request_id, graph_edges in self.tensor_manager.get_ready_tensors().items():
            for graph_edge in graph_edges:
                modality = graph_edge.name.replace("_output", "")

                for tensor_info in graph_edge.tensor_info:
                    logger.debug("Reading in OUTPUT tensor %s with uuid %s", graph_edge.name, tensor_info.uuid)
                    tensor = self.tensor_manager.get_tensor(
                        request_id=request_id,
                        uuid=tensor_info.uuid
                    )
                    postprocessed = self.model.postprocess(
                        tensor, modality
                    )

                    self.out_queue.put(ResultChunk(
                        request_id=request_id,
                        modality=modality,
                        data=postprocessed,
                        metadata=self.tensor_uuid_to_metadata_per_request[request_id][
                            tensor_info.uuid]
                    ))
                    del self.tensor_uuid_to_metadata_per_request[request_id][
                        tensor_info.uuid]
                    self.tensor_manager.dereference(
                        request_id=request_id,
                        uuid=tensor_info.uuid
                    )

    def _process_messages(self):
        for message in self.communicator.get_all_new_messages():
            if message.message_type == WorkerMessageType.TENSOR_RECEIVED:
                body: TensorReceived = message.body
                for (uuid, ref_cnt) in body.successful_tensors.items():
                    self.tensor_manager.dereference(
                        body.request_id, uuid, n=ref_cnt
                    )
            elif message.message_type == WorkerMessageType.UNPERSIST_TENSORS:
                body: UnpersistTensors = message.body
                for (uuid, ref_cnt) in body.uuid_to_ref_count.items():
                    self.tensor_manager.increment_ref(
                        body.request_id, uuid, n=ref_cnt
                    )
                    self.tensor_manager.set_persist(
                        body.request_id, uuid, persist=False
                    )

    def run(self):
        while not self.stop_event.is_set():
            try:
                self._process_messages()
                if not self.in_queue.empty():
                    self._process_input(self.in_queue.get())
                if not self.result_tensor_queue.empty():
                    self._read_result_tensor(self.result_tensor_queue.get())
                if not self.cleanup_request_queue.empty():
                    req_id = self.cleanup_request_queue.get()
                    self.tensor_manager.cleanup_request(req_id)
                    if req_id in self.tensor_uuid_to_metadata_per_request:
                        del self.tensor_uuid_to_metadata_per_request[req_id]
                self._process_read_tensors()
            except Exception:
                logger.exception("PreprocessWorkerThread error")

            time.sleep(0.001)
