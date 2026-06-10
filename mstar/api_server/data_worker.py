

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

from mstar.api_server.request_types import PreprocessInput, ResultChunk, ResultTensors
from mstar.communication.communicator import CommProtocol, ZMQCommunicator
from mstar.communication.tensors import NameToTensorList, create_tensor_communication_manager
from mstar.model.base import Model
from mstar.utils.ipc_format import (
    ConductorMessage,
    ConductorMessageType,
    NewRequestConductor,
    TensorReceived,
    UnpersistTensors,
    WorkerMessageType,
)

logger = logging.getLogger(__name__)

# Lightweight, env-gated timing prints (MMINF_TIMING=1). perf_counter is
# process-wide monotonic, so timestamps stamped in the API-server handler
# thread and read in this data-worker thread are directly comparable — that's
# how queue-wait (polling) latency is separated from actual work below.
_TIMING = os.environ.get("MMINF_TIMING", "") not in ("", "0", "false")


def _tlog(msg: str) -> None:
    if _TIMING:
        print(f"[DW-TIMING] {msg}", flush=True)


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
    ):
        self.request_input_queue = queue.Queue()
        self.result_tensor_input_queue = queue.Queue()
        self.cleanup_request_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.stop_event = threading.Event()

        self.per_request_reading_tensors = {}
        self.output_loop_idxs: dict[str, NameToLoopIndices] = {}

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
                tcp_transfer_device=tcp_transfer_device
            )
        )
        self.thread.start()

    def new_request(self, input: PreprocessInput):
        input._t_enqueue = time.perf_counter()  # for queue-wait timing
        self.output_loop_idxs[input.request_id] = {}
        self.per_request_reading_tensors[input.request_id] = 0
        self.request_input_queue.put(input)

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
        self.cleanup_request_queue.put(request_id)
        del self.output_loop_idxs[request_id]
        del self.per_request_reading_tensors[request_id]

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
        socket_path_prefix: str = "/tmp/mstar",
        tensor_comm_protocol: CommProtocol = CommProtocol.RDMA,
        device: str = "cpu",
        model: Model | None = None,
        tcp_transfer_device="",
    ):
        self.in_queue = in_queue
        self.result_tensor_queue = result_tensor_queue
        self.cleanup_request_queue = cleanup_request_queue
        self.out_queue = out_queue

        self.stop_event = stop_event
        self.device = device
        self.model = model

        self.tensor_uuid_to_metadata_per_request = {}
        self._t_read_start: dict[str, float] = {}  # request_id -> read-start time

        self.communicator = ZMQCommunicator(
            my_id="api_server_preprocess_worker",
            push_ids=["conductor"],
            ipc_socket_path_prefix=socket_path_prefix,
        ) # only used to send

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
        _t0 = time.perf_counter()
        _enq = getattr(input, "_t_enqueue", None)
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

        # ".npy" uploads (modality "numpy") are kept in memory and np.load'd
        # here as "raw_inputs"; the model maps them in process_prompt.
        if input.numpy_bytes:
            import io as _io

            import numpy as np

            tensors["raw_inputs"] = []
            input_metadata["raw_inputs"] = []
            for blob in input.numpy_bytes:
                tensors["raw_inputs"].append(
                    torch.from_numpy(np.load(_io.BytesIO(blob))).to(self.device)
                )
                input_metadata["raw_inputs"].append({})

        _t_load = time.perf_counter()  # media decode (load_image/audio/video) done

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

        _t_prompt = time.perf_counter()  # tokenization / process_prompt done

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

        _t_store = time.perf_counter()  # tensor store/register/persist done

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
        if _TIMING:
            _t_send = time.perf_counter()
            _qwait = (_t0 - _enq) * 1e3 if _enq is not None else -1.0
            _imgs = tensors.get("image_inputs") or []
            _img_shape = tuple(_imgs[0].shape) if _imgs else None
            _tlog(
                f"{input.request_id[:8]} INPUT  "
                f"img={_img_shape}x{len(_imgs)} "  # decoded shape x count (decode cost driver)
                f"qwait={_qwait:.2f} "  # enqueue->dequeue (polling)
                f"load={(_t_load - _t0) * 1e3:.2f} "  # media decode
                f"prompt={(_t_prompt - _t_load) * 1e3:.2f} "  # tokenize
                f"store={(_t_store - _t_prompt) * 1e3:.2f} "  # tensor store/register
                f"send={(_t_send - _t_store) * 1e3:.2f} "  # zmq send to conductor
                f"total={(_t_send - _t0) * 1e3:.2f}ms"
            )

    def _read_result_tensor(
        self, result: ResultTensors
    ):
        self._t_read_start[result.request_id] = time.perf_counter()
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

    def _process_read_tensors(self):
        did_work = False
        for request_id, graph_edges in self.tensor_manager.get_ready_tensors().items():
            did_work = True
            _t_ready = time.perf_counter()  # tensor became ready (RDMA read done)
            _read_start = self._t_read_start.pop(request_id, None)
            for graph_edge in graph_edges:
                modality = graph_edge.name.replace("_output", "")

                for tensor_info in graph_edge.tensor_info:
                    logger.debug("Reading in OUTPUT tensor %s with uuid %s", graph_edge.name, tensor_info.uuid)
                    _t_a = time.perf_counter()
                    tensor = self.tensor_manager.get_tensor(
                        request_id=request_id,
                        uuid=tensor_info.uuid
                    )
                    _t_get = time.perf_counter()
                    postprocessed = self.model.postprocess(
                        tensor, modality
                    )
                    _t_post = time.perf_counter()
                    if _TIMING:
                        _rw = (_t_ready - _read_start) * 1e3 if _read_start else -1.0
                        _tlog(
                            f"{request_id[:8]} OUTPUT "
                            f"read_wait={_rw:.2f} "  # start_read -> ready (RDMA + polling)
                            f"get={(_t_get - _t_a) * 1e3:.2f} "  # fetch tensor handle
                            f"post={(_t_post - _t_get) * 1e3:.2f}ms"  # model.postprocess
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

                    _chunk = ResultChunk(
                        request_id=request_id,
                        modality=modality,
                        data=postprocessed,
                        metadata=chunk_metadata,
                    )
                    _chunk._t_outqueue = time.perf_counter()
                    self.out_queue.put(_chunk)
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

