

from dataclasses import asdict, dataclass
import queue
import threading
import time

import torch
import torchvision
import torchaudio
from torchcodec.decoders import VideoDecoder, AudioDecoder

from mminf.communication.communicator import CommProtocol, ZMQCommunicator
from mminf.communication.tensors import MooncakeCommunicationManager
from mminf.ipc_formats import ConductorMessage, ConductorMessageType, NewRequestConductor


@dataclass
class PreprocessInput:
    request_id: str
    text: str | None

    # file_paths is modality: list of filenames
    file_paths: dict[str, list[str]] | None
    input_modalities: list[str]
    output_modalities: list[str]
    model_kwargs: dict


def _preprocess_loop(**kwargs):
    worker = PreprocessWorkerThread(**kwargs)
    worker.run()


class PreprocessWorker:
    def __init__(
        self,
        hostname: str = "localhost",
        socket_path_prefix: str = "/tmp/mminf",
        tensor_comm_protocol: CommProtocol = CommProtocol.RDMA,
    ):
        self.request_input_queue = queue.Queue()
        self.request_output_queue = queue.Queue()
        self.stop_event = threading.Event()

        self.thread = threading.Thread(
            target=_preprocess_loop,
            kwargs=dict(
                in_queue=self.request_input_queue,
                out_queue=self.request_output_queue,
                stop_event=self.stop_event,
                hostname=hostname,
                socket_path_prefix=socket_path_prefix,
                tensor_comm_protocol=tensor_comm_protocol,
            )
        )
    
    def new_request(self, input: PreprocessInput):
        self.request_input_queue.put(input)
    
    def shutdown(self):
        self.stop_event.set()
        self.thread.join()


class PreprocessWorkerThread:
    def __init__(
        self, 
        in_queue: queue.Queue, # for preprocessing
        out_queue: queue.Queue, # for output streaming
        stop_event: threading.Event,
        hostname: str = "localhost",
        socket_path_prefix: str = "/tmp/mminf",
        tensor_comm_protocol: CommProtocol = CommProtocol.RDMA,
        device: str = "cpu",
    ):
        self.in_queue = in_queue
        self.out_queue = out_queue # unused at the moment
        self.stop_event = stop_event
        self.device = device

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
    

    def process_input(
        self, input: PreprocessInput
    ):
        tensors = {}
        input_metadata = {}
        if input.text is not None:
            # Encode as UTF-8 bytes -> uint8 tensor
            byte_data = input.text.encode("utf-8")
            tensors["text_input"] = torch.tensor(
                list(byte_data), dtype=torch.uint8, device=self.device
            )

        if input.file_paths is not None:
            for modality in input.file_paths:
                for (i, filepath) in input.file_paths[modality]:
                    key = f"{modality}_input_{i}"

                    # ---- Image ----
                    if modality == "image":
                        img = torchvision.io.decode_image(filepath).to(self.device)  # uint8 CxHxW
                        img = img.float() / 255.0
                        tensors[key] = img

                    # ---- Audio ----
                    elif modality == "audio":
                        waveform, sample_rate = torchaudio.load_with_torchcodec(
                            filepath,
                            channels_first=True
                        )
                        # waveform: (channels, time)
                        tensors[key] = waveform
                        input_metadata[key] = dict(
                            sample_rate=sample_rate,
                            channels_first=True
                        )

                    # ---- Video ----
                    elif modality == "video":
                        decoder = VideoDecoder(filepath, device=self.device)
                        video = torch.stack([frame for frame in decoder]).float() / 255.0
                        tensors[key] = video
                        input_metadata[key] = asdict(decoder.metadata)
        
        initial_signals = self.tensor_manager.register_and_return_tensor_info(
            request_id=input.request_id,
            tensors=tensors
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

    def run(self):
        while not self.stop_event.is_set():
            if self.in_queue.not_empty:
                self.process_input(self.in_queue.get())
            time.sleep(0.001)
