

import logging
import queue
import threading
import time

import torch

from mminf.graph.loop_indices import NestedLoopIndices

try:
    import torchaudio  # noqa: F401 — probes availability; real usage in callers
    from torchcodec.decoders import VideoDecoder
except (ImportError, RuntimeError, OSError):
    VideoDecoder = None

from mminf.api_server.request_types import PreprocessInput, ResultChunk, ResultTensors
from mminf.communication.communicator import CommProtocol, ZMQCommunicator
from mminf.communication.event import EventWakeup
from mminf.communication.tensors import NameToTensorList, create_tensor_communication_manager
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


def _postprocess_loop(**kwargs):
    worker = PostprocessWorkerThread(**kwargs)
    worker.run()


NameToLoopIndices = dict[str, NestedLoopIndices]


class DataProcessWorker:
    def __init__(
        self,
        model: Model | None = None,
        hostname: str = "localhost",
        socket_path_prefix: str = "/tmp/mminf",
        tensor_comm_protocol: CommProtocol = CommProtocol.RDMA,
        tcp_transfer_device="",
    ):
        self.request_input_queue = queue.Queue()
        self.result_tensor_input_queue = queue.Queue()
        self.preproc_cleanup_request_queue = queue.Queue()
        self.postproc_cleanup_request_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.preproc_stop_event = threading.Event()
        self.postproc_stop_event = threading.Event()

        self.per_request_reading_tensors = {}
        self.output_loop_idxs: dict[str, NameToLoopIndices] = {}

        # Producers (this api-server thread) put work on the plain queues
        # above; the worker thread blocks in ``wait_for_work`` polling its
        # ZMQ socket. A poll on the socket can't see a queue.put, so signal
        # this eventfd (registered with the worker's poller) on every put to
        # wake the consumer immediately.
        self.preproc_wakeup_event = EventWakeup()
        self.postproc_wakeup_event = EventWakeup()
        # Signalled by the worker thread when it places a result chunk on
        # ``output_queue``; the api server registers this with its own socket
        # poller so its message loop wakes to drain results promptly.
        self.output_wakeup_event = EventWakeup()
        
        self.preprocess_thread = threading.Thread(
            target=_preprocess_loop,
            kwargs=dict(
                in_queue=self.request_input_queue,
                cleanup_request_queue=self.preproc_cleanup_request_queue,
                stop_event=self.preproc_stop_event,
                wakeup_event=self.preproc_wakeup_event,
                hostname=hostname,
                socket_path_prefix=socket_path_prefix,
                tensor_comm_protocol=tensor_comm_protocol,
                model=model,
                tcp_transfer_device=tcp_transfer_device
            )
        )
        self.preprocess_thread.start()

        self.postprocess_thread = threading.Thread(
            target=_postprocess_loop,
            kwargs=dict(
                result_tensor_queue=self.result_tensor_input_queue,
                out_queue=self.output_queue,
                cleanup_request_queue=self.postproc_cleanup_request_queue,
                stop_event=self.postproc_stop_event,
                wakeup_event=self.postproc_wakeup_event,
                output_wakeup_event=self.output_wakeup_event,
                hostname=hostname,
                socket_path_prefix=socket_path_prefix,
                tensor_comm_protocol=tensor_comm_protocol,
                model=model,
                tcp_transfer_device=tcp_transfer_device
            )
        )
        self.postprocess_thread.start()

    def new_request(self, input: PreprocessInput):
        self.output_loop_idxs[input.request_id] = {}
        self.per_request_reading_tensors[input.request_id] = 0
        self.request_input_queue.put(input)
        self.preproc_wakeup_event.signal()

    def new_result_tensors(self, input: ResultTensors):
        name = input.graph_edge.name
        if input.request_id not in self.output_loop_idxs:
            logger.debug("Late result_tensors for cleaned-up request %s, ignoring", input.request_id)
            return

        self.output_loop_idxs[input.request_id][name] = input.loop_indices.max(
            self.output_loop_idxs[input.request_id].get(name, None)
        )

        self.per_request_reading_tensors[input.request_id] += len(input.graph_edge.tensor_info)
        logger.debug(
            "Data worker reading queue for request %s increased to length %d",
            input.request_id,  self.per_request_reading_tensors[input.request_id]
        )
        self.result_tensor_input_queue.put(input)
        self.postproc_wakeup_event.signal()

    def has_pending_tensors(self, request_id: str):
        return self.per_request_reading_tensors.get(request_id, 0) > 0

    def received_final_chunks(
        self, request_id: str,
        final_outputs: dict[str, NestedLoopIndices],
    ):
        return all(
            not loop_iters.label_context_gt( # recv'd loop iters is not less than the final_fwd
                self.output_loop_idxs[request_id].get(name, None)
            ) for name, loop_iters in final_outputs.items()
        )

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
        self.preproc_cleanup_request_queue.put(request_id)
        self.preproc_wakeup_event.signal()
        self.postproc_cleanup_request_queue.put(request_id)
        self.postproc_wakeup_event.signal()
        del self.output_loop_idxs[request_id]
        del self.per_request_reading_tensors[request_id]

    def shutdown(self):
        self.preproc_stop_event.set()
        self.postproc_stop_event.set()
        # Wake both threads out of wait_for_work so they observe the stop
        # events immediately instead of waiting out the poll timeout.
        self.preproc_wakeup_event.signal()
        self.postproc_wakeup_event.signal()
        if self.preprocess_thread.is_alive():
            self.preprocess_thread.join()
        if self.postprocess_thread.is_alive():
            self.postprocess_thread.join()


class PreprocessWorkerThread:
    def __init__(
        self,
        in_queue: queue.Queue, # for preprocessing
        cleanup_request_queue: queue.Queue,
        stop_event: threading.Event,
        wakeup_event: EventWakeup | None = None,
        hostname: str = "localhost",
        socket_path_prefix: str = "/tmp/mminf",
        tensor_comm_protocol: CommProtocol = CommProtocol.RDMA,
        device: str = "cpu",
        model: Model | None = None,
        tcp_transfer_device="",
    ):
        self.in_queue = in_queue
        self.cleanup_request_queue = cleanup_request_queue

        self.stop_event = stop_event
        self.wakeup_event = wakeup_event
        # Signalled when a result chunk is placed on out_queue so the api
        # server's message loop (which polls its own socket) wakes to drain it
        # instead of waiting out a fixed poll interval.
        self.device = device
        self.model = model

        self.communicator = ZMQCommunicator(
            my_id="api_server_preprocess_worker",
            push_ids=["conductor"],
            ipc_socket_path_prefix=socket_path_prefix,
        )
        # Register the producer-side wakeup fd with the socket poller so a
        # single ``wait_for_work`` blocks on both incoming ZMQ messages and
        # new queue work, replacing the old fixed 1ms poll sleep.
        if self.wakeup_event is not None:
            self.communicator.register_event_for_poll(self.wakeup_event)

        self.tensor_manager = create_tensor_communication_manager(
            protocol=tensor_comm_protocol,
            my_entity_id="api_server_preprocess_worker",
            hostname=hostname,
            device=self.device,
            communicator=self.communicator,
            tcp_transfer_device=tcp_transfer_device,
        )

    def _process_input(
        self, input: PreprocessInput
    ):
        tensors: NameToTensorList = {}
        input_metadata = {}

        # First, load raw modality tensors from file_paths (images, audio, video)
        # so they can be passed to process_prompt() below.
        if input.file_paths is not None:
            for modality in input.file_paths:
                key = f"{modality}_inputs"
                tensors[key] = []
                # TODO: maybe make a class of tensors_and_metadata later (figure out how to use metadata)
                input_metadata[key] = []

                for filepath in input.file_paths[modality]:
                    # ---- Image ----
                    if modality == "image":
                        out = self.model.load_image(filepath, self.device)
                        tensors[key].append(out.data)
                        input_metadata[key].append(out.metadata)

                    # ---- Audio ----
                    elif modality == "audio":
                        out = self.model.load_audio(filepath, self.device)
                        tensors[key].append(out.data)
                        input_metadata[key].append(out.metadata)

                    # ---- Video ----
                    elif modality == "video":
                        out = self.model.load_video(filepath, self.device)
                        tensors[key].append(out.data)
                        input_metadata[key].append(out.metadata)


        # Then, tokenize the prompt and let the model augment/transform the
        # tensors dict (e.g., Qwen3-Omni needs to compute pixel_values,
        # image_grid_thw, audio_features, audio_seqlens from the raw tensors
        # loaded above).  process_prompt receives the raw multimodal tensors
        # and returns any additional tensors to merge into the final dict.
        if self.model is not None:
            prompt_tensors = self.model.process_prompt(
                input.text,
                input.input_modalities,
                input.output_modalities,
                tensors=tensors,
                input_metadata=input_metadata,
                **(input.model_kwargs or {}),
            )
            if prompt_tensors:
                tensors.update(prompt_tensors)
        elif input.text is not None:
            # Fallback: encode as UTF-8 bytes -> uint8 tensor
            byte_data = input.text.encode("utf-8")
            tensors["text_inputs"] = [torch.tensor(
                list(byte_data), dtype=torch.uint8, device=self.device
            )]

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

    def _process_messages(self) -> bool:
        """Drain incoming tensor-lifecycle messages. Returns whether any were
        processed (so the interleaved run-loop counts it as progress)."""
        messages = self.communicator.get_all_new_messages()
        for message in messages:
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
        return bool(messages)

    def run(self):
        while not self.stop_event.is_set():
            try:
                progressed = True
                while progressed:
                    progressed = self._process_messages()
                    if not self.in_queue.empty():
                        self._process_input(self.in_queue.get())
                        progressed = True
                    if not self.cleanup_request_queue.empty():
                        req_id = self.cleanup_request_queue.get()
                        self.tensor_manager.cleanup_request(req_id)
                        progressed = True
            except Exception:
                logger.exception("PreprocessWorkerThread error")

            # Block until an incoming ZMQ message, a producer signalling new
            # queue work (eventfd), or a completed async read wakes us —
            # replacing the fixed 1ms sleep that throttled result-tensor
            # ingestion. The timeout is just a safety net for missed wakeups.
            if self.wakeup_event is not None:
                self.communicator.wait_for_work(timeout_ms=10)
            else:
                time.sleep(0.001)


class PostprocessWorkerThread:
    def __init__(
        self,
        result_tensor_queue: queue.Queue, # for output streaming
        out_queue: queue.Queue,
        cleanup_request_queue: queue.Queue,
        stop_event: threading.Event,
        wakeup_event: EventWakeup | None = None,
        output_wakeup_event: EventWakeup | None = None,
        hostname: str = "localhost",
        socket_path_prefix: str = "/tmp/mminf",
        tensor_comm_protocol: CommProtocol = CommProtocol.RDMA,
        device: str = "cpu",
        model: Model | None = None,
        tcp_transfer_device="",
    ):
        self.result_tensor_queue = result_tensor_queue
        self.cleanup_request_queue = cleanup_request_queue
        self.out_queue = out_queue

        self.stop_event = stop_event
        self.wakeup_event = wakeup_event
        # Signalled when a result chunk is placed on out_queue so the api
        # server's message loop (which polls its own socket) wakes to drain it
        # instead of waiting out a fixed poll interval.
        self.output_wakeup_event = output_wakeup_event
        self.device = device
        self.model = model

        self.tensor_uuid_to_metadata_per_request = {}

        self.communicator = ZMQCommunicator(
            my_id="api_server_postprocess_worker",
            push_ids=["conductor"],
            ipc_socket_path_prefix=socket_path_prefix,
        )
        # Register the producer-side wakeup fd with the socket poller so a
        # single ``wait_for_work`` blocks on both incoming ZMQ messages and
        # new queue work, replacing the old fixed 1ms poll sleep.
        if self.wakeup_event is not None:
            self.communicator.register_event_for_poll(self.wakeup_event)

        self.tensor_manager = create_tensor_communication_manager(
            protocol=tensor_comm_protocol,
            my_entity_id="api_server_postprocess_worker",
            hostname=hostname,
            device=self.device,
            communicator=self.communicator,
            tcp_transfer_device=tcp_transfer_device,
        )

    def _process_messages(self) -> bool:
        """Drain incoming tensor-lifecycle messages. Returns whether any were
        processed. The postproc thread is read-only on output tensors and only
        *sends* acks, so it normally receives nothing — but ``wait_for_work``
        polls this socket, and an undrained PULL message stays POLLIN-readable
        and would hot-spin the loop, so drain it defensively (and handle any
        lifecycle message that does route here)."""
        messages = self.communicator.get_all_new_messages()
        for message in messages:
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
        return bool(messages)

    def _read_result_tensor(
        self, result: ResultTensors
    ):
        result.graph_edge.name = f"{result.modality}_output"
        futures = self.tensor_manager.start_read_tensors(
            request_id=result.request_id,
            graph_edges=[result.graph_edge],
        )
        if result.request_id not in self.tensor_uuid_to_metadata_per_request:
            self.tensor_uuid_to_metadata_per_request[result.request_id] = {}
        for tensor_info in result.graph_edge.tensor_info:
            self.tensor_uuid_to_metadata_per_request[result.request_id][
                tensor_info.uuid] = result.metadata
        return futures

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
                    if self.output_wakeup_event is not None:
                        self.output_wakeup_event.signal()
                    del self.tensor_uuid_to_metadata_per_request[request_id][
                        tensor_info.uuid]
                    self.tensor_manager.dereference(
                        request_id=request_id,
                        uuid=tensor_info.uuid
                    )

    def run(self):
        while not self.stop_event.is_set():
            try:
                progressed = True
                while progressed:
                    progressed = self._process_messages()
                    if not self.result_tensor_queue.empty():
                        futures = self._read_result_tensor(self.result_tensor_queue.get())
                        # Async transports (RDMA/TCP) complete reads off-thread;
                        # wake this loop when they land so _process_read_tensors
                        # runs promptly instead of waiting out the poll timeout.
                        if futures and self.wakeup_event is not None:
                            self.wakeup_event.register_futures(futures)
                        progressed = True
                    if not self.cleanup_request_queue.empty():
                        req_id = self.cleanup_request_queue.get()
                        self.tensor_manager.cleanup_request(req_id)
                        if req_id in self.tensor_uuid_to_metadata_per_request:
                            del self.tensor_uuid_to_metadata_per_request[req_id]
                        progressed = True
                    self._process_read_tensors()
            except Exception:
                logger.exception("PostprocessWorkerThread error")

            # Block until an incoming ZMQ message, a producer signalling new
            # queue work (eventfd), or a completed async read wakes us —
            # replacing the fixed 1ms sleep that throttled result-tensor
            # ingestion. The timeout is just a safety net for missed wakeups.
            if self.wakeup_event is not None:
                self.communicator.wait_for_work(timeout_ms=10)
            else:
                time.sleep(0.001)

