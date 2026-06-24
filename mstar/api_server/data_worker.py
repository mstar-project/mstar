

import logging
import os
import queue
import threading
import time

import torch

from mstar.graph.loop_indices import NestedLoopIndices

try:
    import torchaudio  # noqa: F401 — probes availability; real usage in callers
    from torchcodec.decoders import VideoDecoder
except (ImportError, RuntimeError, OSError):
    VideoDecoder = None

from mstar.api_server.request_types import (
    DataWorkerProfile,
    PreprocessInput,
    ResultChunk,
    ResultTensors,
)
from mstar.communication.communicator import CommProtocol, ZMQCommunicator
from mstar.communication.tensors import NameToTensorList, create_tensor_communication_manager
from mstar.model.base import Model
from mstar.profile.format import InputInfo, RxInfo, TxInfo
from mstar.utils.ipc_format import (
    AbortRequest,
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


NameToLoopIndices = dict[str, NestedLoopIndices]


class PreprocessWorker:
    def __init__(
        self,
        model: Model | None = None,
        hostname: str = "localhost",
        socket_path_prefix: str = "/tmp/mstar",
        tensor_comm_protocol: CommProtocol = CommProtocol.RDMA,
        tcp_transfer_device="",
        enable_prof: bool=False
    ):
        self.request_input_queue = queue.Queue()
        self.result_tensor_input_queue = queue.Queue()
        self.cleanup_request_queue = queue.Queue()
        self.abort_request_queue = queue.Queue()
        self.discard_tensor_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.profile_queue = queue.Queue()
        self.stop_event = threading.Event()

        self.per_request_reading_tensors = {}
        self.output_loop_idxs: dict[str, NameToLoopIndices] = {}

        # Build the communicator + tensor manager here (main thread) and hand
        # them to the worker thread, rather than constructing them inside it.
        # The socket is only *used* from the worker thread, but owning the
        # tensor manager here lets the main thread read its tx/rx profiling
        # directly once a request is done (no cross-thread queue / race).
        self.communicator = ZMQCommunicator(
            my_id="api_server_preprocess_worker",
            push_ids=["conductor"],
            ipc_socket_path_prefix=socket_path_prefix,
        )  # only used to send (from the worker thread)
        self.tensor_manager = create_tensor_communication_manager(
            protocol=tensor_comm_protocol,
            my_entity_id="api_server_preprocess_worker",
            hostname=hostname,
            device="cpu",
            communicator=self.communicator,
            tcp_transfer_device=tcp_transfer_device,
            enable_prof=enable_prof,
        )

        self.thread = threading.Thread(
            target=_preprocess_loop,
            kwargs=dict(
                in_queue=self.request_input_queue,
                result_tensor_queue=self.result_tensor_input_queue,
                out_queue=self.output_queue,
                profile_queue=self.profile_queue,
                cleanup_request_queue=self.cleanup_request_queue,
                abort_request_queue=self.abort_request_queue,
                discard_tensor_queue=self.discard_tensor_queue,
                stop_event=self.stop_event,
                communicator=self.communicator,
                tensor_manager=self.tensor_manager,
                model=model,
                enable_prof=enable_prof
            )
        )
        self.thread.start()

    def new_request(self, input: PreprocessInput):
        self.output_loop_idxs[input.request_id] = {}
        self.per_request_reading_tensors[input.request_id] = 0
        self.request_input_queue.put(input)

    def abort_request(self, request_id: str):
        self.abort_request_queue.put(request_id)
        self.cleanup_request(request_id)

    def new_result_tensors(self, input: ResultTensors):
        name = input.graph_edge.name
        if input.request_id not in self.output_loop_idxs:
            # Request was removed while this output was still in flight; ack the
            # tensors so the producing worker can reclaim them rather than leak.
            logger.debug("Late result_tensors for cleaned-up request %s, acking and dropping", input.request_id)
            self.discard_result_tensors(input)
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

    def discard_result_tensors(self, input: ResultTensors):
        """Ack and drop result tensors for an already-removed request.

        Routed to the worker thread (which owns the communicator) so the
        producing worker gets its TENSOR_RECEIVED ack and frees the buffers.
        """
        self.discard_tensor_queue.put(input)

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

    def get_profile_updates(self) -> list[DataWorkerProfile]:
        """Drain preprocess-side profiling updates emitted by the worker thread."""
        updates = []
        while not self.profile_queue.empty():
            updates.append(self.profile_queue.get())
        return updates

    def get_tx_info(self, request_id: str) -> list[TxInfo]:
        """Snapshot the data worker's send (tx) profiling for a request.

        Safe to call from the main thread once the request is done: by then the
        worker thread is no longer mutating this request's tx state, and the
        caller must read it before ``cleanup_request`` drops it.
        """
        return self.tensor_manager.get_tx_info(request_id)

    def get_rx_info(self, request_id: str) -> list[RxInfo]:
        """Snapshot the data worker's receive (rx) profiling for a request.

        Same safety contract as :meth:`get_tx_info` — read once the request's
        final chunks have all arrived, before ``cleanup_request``.
        """
        return self.tensor_manager.get_rx_info(request_id)

    def cleanup_request(self, request_id: str):
        self.cleanup_request_queue.put(request_id)
        self.output_loop_idxs.pop(request_id, None)
        self.per_request_reading_tensors.pop(request_id, None)

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
        profile_queue: queue.Queue,
        cleanup_request_queue: queue.Queue,
        abort_request_queue: queue.Queue,
        discard_tensor_queue: queue.Queue,
        stop_event: threading.Event,
        communicator: ZMQCommunicator,
        tensor_manager,
        device: str = "cpu",
        model: Model | None = None,
        enable_prof: bool=False
    ):
        self.in_queue = in_queue
        self.result_tensor_queue = result_tensor_queue
        self.cleanup_request_queue = cleanup_request_queue
        self.abort_request_queue = abort_request_queue
        self.discard_tensor_queue = discard_tensor_queue
        self.out_queue = out_queue
        self.profile_queue = profile_queue

        self.stop_event = stop_event
        self.device = device
        self.model = model
        self.enable_prof = enable_prof

        self.tensor_uuid_to_metadata_per_request = {}

        # Owned by PreprocessWorker (main thread); used only from this thread.
        self.communicator = communicator
        self.tensor_manager = tensor_manager

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

        # Record preprocess-side profiling: the moment the fully preprocessed
        # request was handed off to the conductor, plus the per-modality sizes of
        # the *raw* inputs. ``perf_counter`` is consistent here because the worker
        # runs as a thread inside the API server process. (tx/rx are snapshotted
        # directly by the main thread at request completion — see APIServer.)
        if self.enable_prof:
            self.profile_queue.put(DataWorkerProfile(
                request_id=input.request_id,
                preprocess_finish_time=time.perf_counter(),
                inputs=self._summarize_inputs(input),
            ))

    @staticmethod
    def _summarize_inputs(input: PreprocessInput) -> list[InputInfo]:
        """Aggregate the *raw* (pre-decoding) inputs into per-modality sizes.

        Reports the bytes the client actually sent — uploaded file sizes on
        disk and the UTF-8 length of the prompt — rather than the much larger
        decoded tensors (e.g. a compressed JPEG vs. its raw RGB tensor), so the
        numbers line up with what a user thinks of as "input size".
        """
        infos = []
        if input.text:
            infos.append(InputInfo(
                modality="text",
                count=1,
                total_bytes=len(input.text.encode("utf-8")),
            ))
        for modality, paths in (input.file_paths or {}).items():
            total_bytes = 0
            for path in paths:
                try:
                    total_bytes += os.path.getsize(path)
                except OSError:
                    pass  # file already cleaned up / unreadable — count as 0
            infos.append(InputInfo(
                modality=modality,
                count=len(paths),
                total_bytes=total_bytes,
            ))
        return infos

    def _read_result_tensor(
        self, result: ResultTensors
    ):
        result.graph_edge.name = f"{result.modality}_output"
        self.tensor_manager.start_read_tensors(
            request_id=result.request_id,
            graph_edges=[result.graph_edge],
        )
        if result.request_id not in self.tensor_uuid_to_metadata_per_request:
            self.tensor_uuid_to_metadata_per_request[result.request_id] = {}
        for tensor_info in result.graph_edge.tensor_info:
            self.tensor_uuid_to_metadata_per_request[result.request_id][
                tensor_info.uuid] = result.metadata

    def _discard_result_tensor(
        self, result: ResultTensors
    ):
        # The request is gone, so don't start a read — just ack the tensors back
        # to the producing worker so it can free the source buffers.
        self.tensor_manager.ack_unread_tensors(
            request_id=result.request_id,
            graph_edges=[result.graph_edge],
        )

    def _process_read_tensors(self):
        did_work = False
        for request_id, graph_edges in self.tensor_manager.get_ready_tensors().items():
            did_work = True
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

                    chunk_metadata = self.tensor_uuid_to_metadata_per_request[request_id][
                        tensor_info.uuid] or {}
                    # Audio is emitted as headerless 16-bit PCM; surface the
                    # model's output sample rate so clients can wrap it.
                    if modality == "audio" and self.model is not None:
                        chunk_metadata = {
                            **chunk_metadata,
                            "sample_rate": self.model.get_output_sample_rate("audio"),
                        }

                    self.out_queue.put(ResultChunk(
                        request_id=request_id,
                        modality=modality,
                        data=postprocessed,
                        metadata=chunk_metadata,
                    ))
                    del self.tensor_uuid_to_metadata_per_request[request_id][
                        tensor_info.uuid]
                    self.tensor_manager.dereference(
                        request_id=request_id,
                        uuid=tensor_info.uuid
                    )
        return did_work

    def _process_messages(self):
        did_work = False
        for message in self.communicator.get_all_new_messages():
            did_work = True
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
        return did_work

    def run(self):
        while not self.stop_event.is_set():
            did_work = False
            try:
                did_work = self._process_messages()
                if not self.in_queue.empty():
                    did_work = True
                    self._process_input(self.in_queue.get())
                if not self.result_tensor_queue.empty():
                    did_work = True
                    self._read_result_tensor(self.result_tensor_queue.get())
                if not self.abort_request_queue.empty():
                    did_work = True
                    self.communicator.send(
                        "conductor",
                        ConductorMessage(
                            message_type=ConductorMessageType.ABORT_REQUEST,
                            body=AbortRequest(request_id=self.abort_request_queue.get()),
                        ),
                    )
                if not self.discard_tensor_queue.empty():
                    did_work = True
                    self._discard_result_tensor(self.discard_tensor_queue.get())
                if not self.cleanup_request_queue.empty():
                    did_work = True
                    req_id = self.cleanup_request_queue.get()
                    self.tensor_manager.cleanup_request(req_id)
                    if req_id in self.tensor_uuid_to_metadata_per_request:
                        del self.tensor_uuid_to_metadata_per_request[req_id]
                did_work = did_work or self._process_read_tensors()
            except Exception:
                logger.exception("PreprocessWorkerThread error")

            if not did_work:
                time.sleep(0.001)

