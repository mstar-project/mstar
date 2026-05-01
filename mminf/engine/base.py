from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

import torch

from mminf.communication.tensors import NameToTensorList
from mminf.conductor.request_info import CurrentForwardPassInfo
from mminf.engine.kv_store import KVCacheConfig


class EngineType(Enum):
    AR = "ar"
    FLOW = "flow"
    ENC_DEC = "enc_dec"
    AUDIO_CODEC = "audio_codec"
    CODE_PREDICTOR = "code_predictor"


@dataclass
class NodeBatch:
    """Input to an engine's execute_batch()."""
    node_name: str
    graph_walk: str
    request_ids: list[str]

    # {request_id: {input_name: list[tensor]}}
    per_request_input_tensors: dict[str, NameToTensorList]
    per_request_info: dict[str, CurrentForwardPassInfo] = field(default_factory=dict)

    # unused for now
    metadata: dict = field(default_factory=dict)


@dataclass
class NodeOutput:
    """Output from an engine's execute_batch()."""
    # {request_id: {output_name: [tensor]}}
    per_request_output_tensors: dict[str, NameToTensorList]
    # Set to True when page allocation failed; worker should hold and retry.
    allocation_failed: bool = False
    # When allocation_failed=True, details about the failure:
    alloc_pages_short: int = 0
    alloc_failed_request_id: str | None = None
    # CUDA event recorded on the default stream after this step's GPU work
    # was submitted, used by the worker to (a) sync only on GPU(N) (not
    # GPU(N+1) which is queued behind it after speculation), and (b) gate a
    # side-stream D→H copy of the produced tokens. Set by the worker in
    # _execute_on_gpu_thread; engines don't populate it themselves.
    completion_event: "torch.cuda.Event | None" = None


class BaseEngine(ABC):
    def __init__(self, enable_nvtx: bool = False, **kwargs):
        self.enable_nvtx = enable_nvtx

    def has_autocast(self):
        return True
    
    def get_max_batch_size(self, node_name: str, graph_walk: str):
        return None

    @abstractmethod
    def engine_type(self) -> EngineType:
        ...

    @abstractmethod
    def load_model(
        self,
        submodules: dict[str, torch.nn.Module],
        kv_cache_config: list[KVCacheConfig],
        device: torch.device,
        **kwargs
    ) -> None:
        """
        Receive the submodules this engine is responsible for
        (keyed by node name) and perform engine-specific initialization
        (KV cache allocation, FlashInfer workspace, etc.).
        """
        ...

    @abstractmethod
    def execute_batch(self, batch: NodeBatch) -> NodeOutput:
        ...

    def execute_with_max_batch_size(self, batch: NodeBatch) -> NodeOutput:
        bs = self.get_max_batch_size(batch.node_name, batch.graph_walk)
        n = len(batch.request_ids)
        if bs is None or n <= bs:
            return self.execute_batch(batch)

        output = NodeOutput(
            per_request_output_tensors={}
        )

        for i in range(0, n, bs):
            rids = batch.request_ids[i:min(i+bs, n)]
            minibatch_out = self.execute_batch(NodeBatch(
                node_name=batch.node_name,
                graph_walk=batch.graph_walk,
                request_ids=rids,
                per_request_input_tensors={
                    rid: batch.per_request_input_tensors[rid] for rid in rids \
                        if rid in batch.per_request_input_tensors
                },
                per_request_info={
                    rid: batch.per_request_info[rid] for rid in rids \
                        if rid in batch.per_request_info
                },
                metadata=batch.metadata
            ))
            output.per_request_output_tensors.update(
                minibatch_out.per_request_output_tensors
            )
            if minibatch_out.allocation_failed:
                output.allocation_failed = True
                output.alloc_pages_short = minibatch_out.alloc_pages_short
                output.alloc_failed_request_id = minibatch_out.alloc_failed_request_id
                return output
        return output

    @abstractmethod
    def add_request(self, request_id: str, **kwargs) -> None:
        ...

    @abstractmethod
    def remove_request(self, request_id: str) -> None:
        ...

    def check_ready(
        self, node_name: str, request_id: str,
        request_info: CurrentForwardPassInfo,
    ):
        """
        Check if the engine is ready to execute.
        """
        return True

    def check_stop_for_batch(
        self, batch: NodeBatch, output: NodeOutput
    ) -> dict[str, set[str]]:
        """
        Per-rid stop-condition check for a finished batch.

        Called by the worker on its slow-postprocess path *after*
        ``execute_batch`` returns. May read tensor values. Returns
        ``{request_id: {loop_name, ...}}`` for rids whose loops should stop.

        Default: no stops (engines without value-driven stop conditions —
        FlowEngine, EncoderDecoderEngine, AudioCodecEngine — return {}).
        """
        return {}

    def warmup(self) -> None:
        """Optional CUDA graph capture. Override in subclasses."""
        return

    def shutdown(self) -> None:
        return
