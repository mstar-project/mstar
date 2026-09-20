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


def _handed_over(model_kwargs: dict, key: str = "kv") -> KVReqConfig:
    """The config one resource ends up with, as the conductor's loop builds it."""
    cfg = KVReqConfig()
    cfg.apply_conductor_config(
        seed=1,
        prefix_keys=(model_kwargs.get("prefix_keys") or {}).get(key),
        prefix_tail=(model_kwargs.get("prefix_tail") or {}).get(key),
        prefix_decode=(model_kwargs.get("prefix_decode") or {}).get(key),
        prefix_cache=model_kwargs.get("prefix_cache"),
    )
    return cfg


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


# ── keys a client sent ──────────────────────────────────────────────────

_PLANTED = {
    "prefix_keys": {"kv": {"main": [b"another request's page"]}},
    "prefix_tail": {"kv": {"main": [7, 7, 7]}},
    "prefix_decode": {"kv": {"main": "text_inputs"}},
}


def test_keys_a_client_sent_never_reach_an_undeclared_models_cache():
    worker = _worker(_Model(declares=False), _deployment())

    cfg = _handed_over(_run(worker, dict(_PLANTED)))

    assert (cfg.prefix_keys, cfg.prefix_tail, cfg.prefix_decode) == (None, None, None), (
        "a request named pages by keys it made up, and would attend whatever "
        "another request left under them"
    )


def test_a_declared_model_is_keyed_by_its_own_prompt_not_by_the_client():
    worker = _worker(_Model(), _deployment())

    cfg = _handed_over(_run(worker, dict(_PLANTED)))

    assert cfg.prefix_keys == {"main": chain([
        PROMPT[at:at + PAGE_SIZE] for at in range(0, len(PROMPT), PAGE_SIZE)
    ])}, "the client's keys survived beside the ones the worker chained"
    assert cfg.prefix_tail == {"main": PROMPT[len(PROMPT) // PAGE_SIZE * PAGE_SIZE:]}, (
        "the client's tail replaced the prompt's own"
    )
    assert cfg.prefix_decode is None, (
        "the client made a node chain its generation that never declared it"
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


# ── one request opting out ──────────────────────────────────────────────


def test_a_request_can_turn_the_cache_off_for_itself():
    configs = {"kv": KVReqConfig()}

    # what the conductor does with a request that sent prefix_cache=False
    for cfg in configs.values():
        cfg.apply_conductor_config(seed=1, prefix_cache=False)

    assert configs["kv"].prefix_cache is False


def test_a_request_that_says_nothing_is_still_cached():
    cfg = KVReqConfig()

    cfg.apply_conductor_config(seed=1, prefix_keys={"main": [b"k"]})

    assert cfg.prefix_cache is True, (
        "a request that never mentioned the cache was opted out of it"
    )


def test_a_request_that_opted_out_keys_no_stream_at_ingest():
    import torch

    from mstar.engine.resources.kv import manager as kv_manager_mod
    from mstar.engine.resources.kv.manager import KVManager

    class _NoTransfer:
        def __init__(self, *a, **k):
            pass

        def get_kv_transfer_info(self):
            return None

        def cleanup(self):
            pass

    original = kv_manager_mod.KVTransferManager
    kv_manager_mod.KVTransferManager = _NoTransfer
    try:
        kv = KVManager(
            cfg=KVConfig(
                num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=64,
                max_num_pages=16, page_size=PAGE_SIZE,
            ),
            name="kv", joint_comm_group=None, transfer_engine_info=None,
            device=torch.device("cpu"), dtype=torch.float32,
        )
        kv.enable_prefix_cache(b"root")
        cfg = KVReqConfig(prefix_keys={"main": [b"k0"]})
        cfg.apply_conductor_config(seed=1, prefix_cache=False)

        kv.ingest_request("r0", cfg)

        assert kv._streams["r0"]["main"].keys is None, (
            "the opt-out reached the config but not the stream"
        )
    finally:
        kv_manager_mod.KVTransferManager = original
