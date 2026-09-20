"""Prefix keys ride the request record, chained where the tokens already are.

The KV resource never sees tokens, and the prompt's ids sit on the host only
before the copy to device, so the keys that name a prompt's pages are chained on
the preprocess worker and travel to the manager as ordinary request metadata.
Paging them needs the deployment's page size, not the model's default, which is
why the worker resolves the specs the same way the engine does.
"""

from __future__ import annotations

import queue
import sys
import threading

sys.path.insert(0, ".")

import torch

from mstar.api_server.data_worker import PreprocessWorkerThread
from mstar.api_server.request_types import PreprocessInput
from mstar.engine.resources.kv.config import KVConfig, KVReqConfig, KVSpec
from mstar.engine.resources.kv.keys import chain
from mstar.model.base import PrefixStream, ProcessPromptOutput

PAGE_SIZE = 16
PROMPT = list(range(100))


class _RecordingCommunicator:
    """Keeps what the worker sent the conductor."""

    def __init__(self):
        self.sent = []

    def get_all_new_messages(self):
        return []

    def send(self, entity, msg):
        self.sent.append(msg)


class _StubTensorManager:
    """Stores no tensor; the request never reaches a device."""

    def store_and_return_tensor_info(self, request_id, tensors):
        return {}

    def register_for_send(self, request_id, tensor_infos):
        pass

    def set_persist(self, request_id, uuid, persist):
        pass


class _Model:
    """A model that declares one id-keyed stream, or none when asked not to."""

    def __init__(self, declares: bool = True, metadata: dict | None = None):
        self._declares = declares
        self._metadata = metadata

    def prefix_key_streams(self):
        if not self._declares:
            return {}
        return {"kv": {"main": PrefixStream("text_inputs", "ids")}}

    def get_node_resources(self):
        return [KVSpec(
            resource_key="kv", nodes={"LLM"},
            config=KVConfig(
                num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=4096,
                # the deployment overrides this; a key paged at 128 would
                # never match one the engine paged at PAGE_SIZE
                max_num_pages=64, page_size=128,
            ),
        )]

    def process_prompt(self, *args, **kwargs):
        tensors = {"text_inputs": [torch.tensor(PROMPT, dtype=torch.long)]}
        if self._metadata is None:
            return tensors
        return ProcessPromptOutput(tensors, self._metadata)


def _worker(model, model_config: dict | None) -> PreprocessWorkerThread:
    return PreprocessWorkerThread(
        in_queue=queue.Queue(), result_tensor_queue=queue.Queue(),
        out_queue=queue.Queue(), profile_queue=queue.Queue(),
        cleanup_request_queue=queue.Queue(), abort_request_queue=queue.Queue(),
        reads_done_queue=queue.Queue(), discard_tensor_queue=queue.Queue(),
        stop_event=threading.Event(), communicator=_RecordingCommunicator(),
        tensor_manager=_StubTensorManager(), model=model,
        model_config=model_config,
    )


def _deployment(page_size: int = PAGE_SIZE) -> dict:
    return {"model": "stub", "resources": {"kv": {"page_size": page_size}}}


def _run(worker: PreprocessWorkerThread, model_kwargs: dict | None = None) -> dict:
    """Preprocess one request and return the kwargs the conductor was sent."""
    worker._process_input(PreprocessInput(
        request_id="r0", text="hello", file_paths=None,
        input_modalities=["text"], output_modalities=["text"],
        model_kwargs=model_kwargs or {},
    ))
    return worker.communicator.sent[-1].body.model_kwargs


# ── what the worker chains ──────────────────────────────────────────────


def test_a_declared_stream_arrives_as_one_key_a_page():
    worker = _worker(_Model(), _deployment())

    keys = _run(worker)["prefix_keys"]["kv"]["main"]

    pages = -(-len(PROMPT) // PAGE_SIZE)
    assert len(keys) == pages, f"{len(PROMPT)} tokens keyed into {len(keys)} pages"
    assert keys == chain([
        PROMPT[at:at + PAGE_SIZE] for at in range(0, len(PROMPT), PAGE_SIZE)
    ]), "the worker's chain is not the one the manager will recompute"


def test_the_deployments_page_size_is_the_one_that_pages_the_stream():
    worker = _worker(_Model(), _deployment(page_size=32))

    keys = _run(worker)["prefix_keys"]["kv"]["main"]

    assert len(keys) == -(-len(PROMPT) // 32), (
        "the model's declared page size was used instead of the deployment's"
    )


def test_an_undeclared_model_sends_no_keys():
    worker = _worker(_Model(declares=False), _deployment())

    assert "prefix_keys" not in _run(worker), "an undeclared model was keyed"


def test_a_worker_without_a_deployment_config_keys_no_stream():
    worker = _worker(_Model(), None)

    assert "prefix_keys" not in _run(worker), (
        "a stream was keyed without a page size to page it by"
    )


# ── the metadata fold ───────────────────────────────────────────────────


def test_the_metadata_fold_leaves_the_requests_own_kwargs_alone():
    worker = _worker(_Model(metadata={"from_process_prompt": 1}), _deployment())

    kwargs = _run(worker, {"voice": "tara"})

    assert kwargs["voice"] == "tara", "the fold dropped the request's own kwargs"
    assert kwargs["from_process_prompt"] == 1, "the metadata never arrived"


def test_a_model_returning_only_tensors_still_carries_its_keys():
    worker = _worker(_Model(metadata=None), _deployment())

    kwargs = _run(worker, {"voice": "tara"})

    assert kwargs["voice"] == "tara"
    assert "prefix_keys" in kwargs, "the plain-dict return lost the keys"


# ── the handover to the resource config ─────────────────────────────────


def test_each_resource_config_is_handed_only_its_own_chain():
    worker = _worker(_Model(), _deployment())
    prefix_keys = _run(worker)["prefix_keys"]
    configs = {"kv": KVReqConfig(), "other_kv": KVReqConfig()}

    # what the conductor does once the configs are resolved
    for key, cfg in configs.items():
        cfg.apply_conductor_config(seed=1, prefix_keys=prefix_keys.get(key))

    assert configs["kv"].prefix_keys == prefix_keys["kv"]
    assert configs["other_kv"].prefix_keys is None, (
        "a resource was handed another resource's chain"
    )


def test_a_request_config_is_cached_until_it_says_otherwise():
    assert KVReqConfig().prefix_cache is True
