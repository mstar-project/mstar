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
from types import SimpleNamespace

sys.path.insert(0, ".")

import torch

from mstar.api_server.data_worker import PreprocessWorkerThread
from mstar.api_server.request_types import PreprocessInput
from mstar.conductor.conductor import Conductor
from mstar.engine.resources import SamplingReqConfig
from mstar.engine.resources.kv.config import KVReqConfig, KVSpec, PagedKVConfig
from mstar.engine.resources.kv.keys import chain
from mstar.model.base import PrefixStream, ProcessPromptOutput
from mstar.model.orpheus.config import OrpheusModelConfig
from mstar.model.orpheus.orpheus_model import OrpheusModel

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

    def __init__(
        self, declares: bool = True, metadata: dict | None = None,
        decode_walk: str | None = None,
    ):
        self._declares = declares
        self._metadata = metadata
        self._decode_walk = decode_walk

    def prefix_key_streams(self):
        if not self._declares:
            return {}
        return {"kv": {"main": PrefixStream(
            "text_inputs", "ids", "prefill", self._decode_walk,
        )}}

    def get_node_resources(self):
        return [KVSpec(
            resource_key="kv", nodes={"LLM"},
            config=PagedKVConfig(
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


def _apply_chain(cfg, key: str, model_kwargs: dict | None) -> None:
    """One config's share of the conductor's loop over a request's chains."""
    kwargs = model_kwargs or {}
    cfg.apply_conductor_config(
        seed=1,
        prefix_keys=(kwargs.get("prefix_keys") or {}).get(key),
        prefix_tail=(kwargs.get("prefix_tail") or {}).get(key),
        prefix_decode=(kwargs.get("prefix_decode") or {}).get(key),
        prefix_cache=kwargs.get("prefix_cache"),
    )


def _handed_over(model_kwargs: dict, key: str = "kv") -> KVReqConfig:
    """The config one resource ends up with, as the conductor's loop builds it."""
    cfg = KVReqConfig()
    _apply_chain(cfg, key, model_kwargs)
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


def test_the_tail_the_manager_cannot_see_travels_with_the_keys():
    worker = _worker(_Model(), _deployment())

    cfg = _handed_over(_run(worker))

    whole = len(PROMPT) // PAGE_SIZE
    assert cfg.prefix_tail == {"main": PROMPT[whole * PAGE_SIZE:]}, (
        "the page the generation finishes is keyed over the prompt's last "
        "tokens, and the manager has no other way to see them"
    )


def test_a_node_that_chains_its_decode_says_where_the_token_arrives():
    worker = _worker(_Model(decode_walk="decode"), _deployment())

    cfg = _handed_over(_run(worker))

    assert cfg.prefix_decode == {"main": "text_inputs"}, (
        "the manager was not told which output carries the sampled token"
    )


def test_a_node_that_does_not_chain_its_decode_sends_no_name():
    worker = _worker(_Model(decode_walk=None), _deployment())

    cfg = _handed_over(_run(worker))

    assert cfg.prefix_decode is None, (
        "a node that never said its decode ids are the token it sampled would "
        "have its generation keyed anyway"
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


class _Recording(_Model):
    """Keeps the keyword arguments its prompt processing was handed."""

    def process_prompt(self, *args, **kwargs):
        self.handed = kwargs
        return super().process_prompt(*args, **kwargs)


def test_a_clients_prefix_keys_never_reach_process_prompt():
    model = _Recording()
    worker = _worker(model, _deployment())

    _run(worker, dict(_PLANTED))

    assert not {"prefix_keys", "prefix_tail", "prefix_decode"} & set(model.handed), (
        "process_prompt was handed keys the client made up, and a model that "
        f"reads them would key the prompt by them: {sorted(model.handed)}"
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

    assert kwargs["voice"] == "tara", "the request's own kwargs were dropped"
    assert "prefix_keys" in kwargs, "the plain-dict return lost the keys"


# ── the handover to the resource config ─────────────────────────────────


def test_each_resource_config_is_handed_only_its_own_chain():
    worker = _worker(_Model(), _deployment())
    prefix_keys = _run(worker)["prefix_keys"]
    configs = {"kv": KVReqConfig(), "other_kv": KVReqConfig()}

    # what the conductor does once the configs are resolved
    for key, cfg in configs.items():
        cfg.apply_conductor_config(seed=1, prefix_keys=prefix_keys.get(key))

    assert configs["kv"].prefix_keys == prefix_keys["kv"], (
        "the resource that keyed this stream was not handed its chain"
    )
    assert configs["other_kv"].prefix_keys is None, (
        "a resource was handed another resource's chain"
    )


# ── one request opting out ──────────────────────────────────────────────


def test_a_request_can_turn_the_cache_off_for_itself():
    configs = {"kv": KVReqConfig()}

    # what the conductor does with a request that sent prefix_cache=False
    for cfg in configs.values():
        cfg.apply_conductor_config(seed=1, prefix_cache=False)

    assert configs["kv"].prefix_cache is False, (
        "a request that asked for no cache would still be matched"
    )


def test_a_request_that_says_nothing_is_still_cached():
    cfg = KVReqConfig()

    cfg.apply_conductor_config(seed=1, prefix_keys={"main": [b"k"]})

    assert cfg.prefix_cache is True, (
        "a request that never mentioned the cache was opted out of it"
    )


# ── a config for every resource that declared a stream ──────────────────


class _SamplerOnlyModel(_Model):
    """Declares a stream and asks for no config of its own, as Orpheus did."""

    def get_request_resource_configs(self, partition_fwd_args, model_kwargs=None):
        del partition_fwd_args, model_kwargs
        return {"sampler": SamplingReqConfig()}


def _conductor_configs(model, model_kwargs: dict | None = None) -> dict:
    """The configs a request is opened with, as the conductor resolves them:
    the model's own, a default for anything it declared a stream for, and then
    each handed its own chain."""
    configs = Conductor._get_resource_configs(
        SimpleNamespace(model=model), model_kwargs, {},
    )
    for key, cfg in configs.items():
        _apply_chain(cfg, key, model_kwargs)
    return configs


def test_a_model_returning_only_a_sampler_config_still_gets_its_keys():
    model = _SamplerOnlyModel()
    model_kwargs = _run(_worker(model, _deployment()))

    handed = _conductor_configs(model, model_kwargs).get("kv")

    assert handed is not None and handed.prefix_keys == model_kwargs["prefix_keys"]["kv"], (
        "the resource that declared the stream was handed no config of its "
        "own, so the chain the worker built for it went nowhere"
    )


def test_orpheus_is_handed_a_config_for_the_stream_it_declares():
    orpheus = OrpheusModel.__new__(OrpheusModel)
    orpheus.config = OrpheusModelConfig()

    configs = _conductor_configs(orpheus)

    assert set(orpheus.prefix_key_streams()) <= set(configs), (
        "orpheus returns a sampler config alone, so the resource it declared "
        "a stream for had nothing to carry its chain"
    )
