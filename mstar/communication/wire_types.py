"""Every class that may cross a communicator edge, registered by stable tag.

Importing this module is what makes ``mstar.communication.wire`` able to
encode. It is kept apart from ``wire.py`` so the codec has no dependency on
the message definitions (and so the import cycle runs one way).

A tag is part of the wire format: rename a class freely, but changing its tag
breaks compatibility with a peer running the old build.
"""
from mstar.api_server.request_types import (
    APIServerMessage,
    DataWorkerProfile,
    PreprocessInput,
    PromptPart,
    RequestComplete,
    RequestFailed,
    ResultChunk,
    ResultTensors,
)
from mstar.communication.wire import _set_polymorphic, register
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.resources.base import PublishedInfo
from mstar.engine.resources.kv.config import KVReqConfig
from mstar.engine.resources.kv.manager import PublishedKVInfo
from mstar.engine.resources.kv.transfer import CudaIpcKVTransferInfo
from mstar.engine.resources.position.manager import PublishedPositionInfo
from mstar.engine.resources.sampler.config import SamplingReqConfig
from mstar.engine.resources.spec import ResourceReqConfig
from mstar.utils.ipc_format import (
    AbortRequest,
    ConductorMessage,
    DrainRequest,
    FailRequests,
    InputSignals,
    MessageBody,
    NewRequest,
    NewRequestConductor,
    ReadsDone,
    RemoveRequest,
    ScheduleTPNode,
    SetupDone,
    StopLoops,
    TensorReceived,
    TPNoSpeculation,
    UnpersistTensors,
    WorkerGraphsDone,
    WorkerMessage,
)

# The three abstract field types; a value of one of these carries [tag, body].
_set_polymorphic(MessageBody, PublishedInfo, ResourceReqConfig)

_TYPES: dict[str, type] = {
    # top-level envelopes
    "worker_msg": WorkerMessage,
    "conductor_msg": ConductorMessage,
    "api_msg": APIServerMessage,
    # worker-bound bodies
    "new_request": NewRequest,
    "remove_request": RemoveRequest,
    "drain_request": DrainRequest,
    "input_signals": InputSignals,
    "tensor_received": TensorReceived,
    "unpersist_tensors": UnpersistTensors,
    "stop_loops": StopLoops,
    "schedule_tp_node": ScheduleTPNode,
    "tp_no_speculation": TPNoSpeculation,
    # conductor-bound bodies
    "wgs_done": WorkerGraphsDone,
    "setup_done": SetupDone,
    "abort_request": AbortRequest,
    "new_request_conductor": NewRequestConductor,
    "reads_done": ReadsDone,
    "fail_requests": FailRequests,
    # api-server bodies
    "result_tensors": ResultTensors,
    "request_complete": RequestComplete,
    "request_failed": RequestFailed,
    "result_chunk": ResultChunk,
    "preprocess_input": PreprocessInput,
    "data_worker_profile": DataWorkerProfile,
    "prompt_part": PromptPart,
    # polymorphic leaves
    "fwd_pass_info": CurrentForwardPassInfo,
    "published_kv": PublishedKVInfo,
    "cuda_ipc_kv_transfer": CudaIpcKVTransferInfo,
    "published_position": PublishedPositionInfo,
    "kv_req_config": KVReqConfig,
    "sampling_req_config": SamplingReqConfig,
}

for _tag, _cls in _TYPES.items():
    register(_tag, _cls)
