"""A request its KV pool can never hold is answered with a 400 at ingest.

The pool fails such a request at its first readiness ask, on the worker, and the
client heard of it as a 500 naming the resource, after workers had been sent a
request that never could run. The conductor knows the request's count and the
deployment's pool when it ingests the request, so it refuses it there, before
any worker hears of it, and tells the client what to shorten.

The request is preprocessed by the data worker, as the API sends it, so its
count is the one the conductor is really handed.
"""

from __future__ import annotations

import queue
import threading
import types
from collections import deque

import torch

from mstar.api_server.data_worker import PreprocessWorkerThread
from mstar.api_server.request_types import PreprocessInput
from mstar.conductor.conductor import Conductor
from mstar.conductor.request_info import CurrentForwardConductorMetadata
from mstar.engine.resources.kv.config import KVReqConfig, KVSpec, PagedKVConfig
from mstar.model.base import ForwardPassArgs, Model
from mstar.utils.ipc_format import ReadsDone, WorkerMessageType

PREPROCESS = "api_server_preprocess_worker"
PAGE_SIZE = 16
# a pool of 8 pages, the sink aside: 7 pages of 16 tokens for requests
DEPLOYMENT = {"resources": {"kv": {"max_num_pages": 8, "page_size": PAGE_SIZE}}}


class _StubTensorManager:
    """Stores no tensor; the request never reaches a device."""

    def store_and_return_tensor_info(self, request_id, tensors, **kwargs):
        return {}

    def register_for_send(self, request_id, tensor_infos, **kwargs):
        pass

    def set_persist(self, uuid, persist):
        pass


class _Model(Model):
    """One prefill walk on one node, its pool counted a slot per prompt id, and decode."""

    def kv_slots(self, tensors, input_modalities, output_modalities, **kwargs):
        return {
            "prompt_slots": {"kv": {"main": len(tensors["text_inputs"][0])}},
            "decode_labels": {"kv": ["main"]},
        }

    def kv_counted(self):
        return frozenset({"kv"})

    def get_node_resources(self):
        return [KVSpec(
            resource_key="kv", nodes={"LLM"},
            config=PagedKVConfig(num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=4096),
        )]

    def get_graph_walk_graphs(self):
        return {"prefill": None}

    def get_initial_forward_pass_args(
        self, partition_name, input_modalities, output_modalities, input_signals, model_kwargs=None,
    ):
        return ForwardPassArgs(
            full_metadata=CurrentForwardConductorMetadata(
                input_modalities=input_modalities, output_modalities=output_modalities,
                graph_walk="prefill", is_prefill=True,
            ),
            inputs=[], unpersist_tensors=[],
        )

    def get_partition_forward_pass_args(self, *args, **kwargs):
        raise NotImplementedError

    def process_prompt(self, prompt, input_modalities, output_modalities, tensors=None, **kwargs):
        return {"text_inputs": [torch.tensor(list(prompt.encode()), dtype=torch.long)]}

    def postprocess(self, output, modality, request_kwargs=None):
        raise NotImplementedError

    def get_submodule(self, node_name, device="cpu", tp_group=None, autocast_dtype=None, sp_group=None):
        return None


class _GuidedModel(_Model):
    """Bagel's shape: a guidance branch on a second worker shares the pool and
    writes a label of its own there, which the leader's instance never holds."""

    def get_node_resources(self):
        (spec,) = super().get_node_resources()
        spec.nodes, spec.leader = {"LLM", "LLM_cfg"}, "LLM"
        return [spec]

    def get_request_resource_configs(self, partition_fwd_args, model_kwargs=None):
        return {"kv": KVReqConfig(needed_labels_per_node={"LLM_cfg": ["cfg"]})}

    def kv_slots(self, tensors, input_modalities, output_modalities, **kwargs):
        slots = super().kv_slots(tensors, input_modalities, output_modalities)
        slots["prompt_slots"]["kv"]["cfg"] = slots["prompt_slots"]["kv"]["main"]
        return slots


class _PartlyCountedModel(_Model):
    """A second label on the pool's own node that the model neither counts nor keys."""

    def get_request_resource_configs(self, partition_fwd_args, model_kwargs=None):
        return {"kv": KVReqConfig(needed_labels_per_node={"LLM": ["main", "uncounted"]})}


def _preprocessed(model: Model, prompt: str, max_output_tokens: int):
    """The NEW_REQUEST the data worker sends the conductor for one text request."""
    sent = []
    worker = PreprocessWorkerThread(
        in_queue=queue.Queue(), result_tensor_queue=queue.Queue(),
        out_queue=queue.Queue(), profile_queue=queue.Queue(),
        cleanup_request_queue=queue.Queue(), abort_request_queue=queue.Queue(),
        reads_done_queue=queue.Queue(), discard_tensor_queue=queue.Queue(),
        stop_event=threading.Event(),
        communicator=types.SimpleNamespace(send=lambda entity, msg: sent.append(msg)),
        tensor_manager=_StubTensorManager(), model=model, model_config=DEPLOYMENT,
    )
    worker._process_input(PreprocessInput(
        request_id="r0", text=prompt, file_paths=None,
        input_modalities=["text"], output_modalities=["text"],
        model_kwargs={"max_output_tokens": max_output_tokens},
    ))
    return sent[-1].body


def _conductor(model: Model, worker_nodes: dict[str, set[str]] | None = None) -> Conductor:
    """A conductor over workers hosting the model's nodes, one by default, sending nowhere."""
    worker_nodes = worker_nodes or {"worker_0": {"LLM"}}
    c = Conductor.__new__(Conductor)
    c.sent = []
    c.communicator = types.SimpleNamespace(send=lambda entity, msg: c.sent.append((entity, msg)))
    c.model = model
    c.model_config = DEPLOYMENT
    c.requests = {}
    c.draining = {}
    c.waiting_queue = []
    c.max_concurrent_requests = None
    c.enable_nvtx = False
    c._drain_ttl_s = 120.0
    c._early_reads_done = {}
    c._early_abort_requests = set()
    c._draining_deadlines = deque()
    c._max_loop_iters = 4096
    c.worker_ids = list(worker_nodes)
    c.worker_graphs = {0: types.SimpleNamespace(graph_walks={"prefill"})}
    c._all_worker_graph_ids_to_graph_walks = {0: {"prefill"}}
    c._worker_nodes = worker_nodes.get
    c._assign_worker_graphs_to_workers = lambda: {0: ["worker_0"]}
    c._build_request_sharding_config = lambda worker_graph_to_workers: None
    c._split_inputs_to_workers = lambda **kwargs: {}
    c._try_admit_waiting = lambda: None
    c._kv_cache_keys = c._counted_kv_keys(model)
    return c


def _sent(c: Conductor, entity: str, message_type):
    return [m for e, m in c.sent if e == entity and m.message_type == message_type]


def test_a_request_its_pool_can_never_hold_is_answered_with_a_400_naming_the_limit():
    model = _Model()
    c = _conductor(model)

    # 20 prompt ids and 200 decoded, past the 7 pages of 16 the pool has
    c._ingest_request(_preprocessed(model, "x" * 20, max_output_tokens=200))
    assert not _sent(c, "worker_0", WorkerMessageType.NEW_REQUEST), "a worker was sent a request that cannot run"
    c._handle_reads_done(ReadsDone(request_id="r0", entity_id=PREPROCESS))

    (failed,) = _sent(c, "api_server", "request_failed")
    assert failed.body.status == 400, "a request too big for the pool was answered as a server error"
    assert failed.body.error_message == (
        "KV pool 'kv' cannot hold this request: it needs 14 pages of 16 tokens, and the pool holds 7 "
        "for requests. To fit, send a shorter prompt, fewer or smaller attachments or a lower max_tokens; "
        "or serve it with a larger resources.kv.max_num_pages"
    ), "the 400 does not name the pool, its limit and what to change"
    assert "r0" not in c.requests


def test_a_request_its_pool_can_hold_goes_to_its_worker():
    model = _Model()
    c = _conductor(model)

    c._ingest_request(_preprocessed(model, "x" * 20, max_output_tokens=60))

    assert _sent(c, "worker_0", WorkerMessageType.NEW_REQUEST) and "r0" not in c.draining, (
        "a request that fits the pool was refused"
    )


def test_a_pool_spanning_workers_is_counted_as_its_leader_holds_it():
    model = _GuidedModel()
    c = _conductor(model, {"worker_0": {"LLM"}, "worker_1": {"LLM_cfg"}})

    # 6 pages on the leader's instance; 8 with the guidance label it never holds
    c._ingest_request(_preprocessed(model, "x" * 20, max_output_tokens=60))

    assert _sent(c, "worker_0", WorkerMessageType.NEW_REQUEST) and "r0" not in c.draining, (
        "the guidance branch's label was counted against the leader's pool"
    )


def test_a_label_the_model_does_not_count_is_refused_at_ingest_naming_it():
    model = _PartlyCountedModel()
    c = _conductor(model)

    # it would run with no reservation, since the pool reserves only once every label is counted
    c._ingest_request(_preprocessed(model, "x" * 20, max_output_tokens=60))
    assert not _sent(c, "worker_0", WorkerMessageType.NEW_REQUEST), "a worker was sent a request no count covers"
    c._handle_reads_done(ReadsDone(request_id="r0", entity_id=PREPROCESS))

    (failed,) = _sent(c, "api_server", "request_failed")
    assert failed.body.error_message == (
        "_PartlyCountedModel's kv_slots counts nothing for label 'uncounted' of KV pool 'kv', which this request opens"
    ), "the refusal does not name the model and the label it left uncounted"


def test_a_request_decoding_to_its_loops_last_iteration_fits_where_its_walk_fits():
    model = _Model()
    c = _conductor(model)
    c._max_loop_iters = 92

    # 20 prompt ids and the loop's 92 iterations, none speculated past the last: 112 slots, all 7 pages
    c._ingest_request(_preprocessed(model, "x" * 20, max_output_tokens=200))

    assert _sent(c, "worker_0", WorkerMessageType.NEW_REQUEST) and "r0" not in c.draining, (
        "a step past the loop's last iteration was counted, refusing a request whose walk fits the pool"
    )
