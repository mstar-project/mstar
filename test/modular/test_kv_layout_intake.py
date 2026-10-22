"""A request's KV layout travels from `process_prompt` to the stream that holds its pages."""

from __future__ import annotations

import queue
import sys
import threading
from collections import deque

sys.path.insert(0, ".")

import pytest
import torch
import yaml
from test_kv_prefix_request_keys import _StubTensorManager

import mstar.conductor.conductor as conductor_mod
from mstar.api_server.data_worker import PreprocessWorkerThread
from mstar.api_server.request_types import PreprocessInput
from mstar.communication.tensors import LocalTransferEngine
from mstar.conductor.conductor import Conductor
from mstar.conductor.request_info import CurrentForwardConductorMetadata
from mstar.engine.resources.kv.config import KVReqConfig, KVSpec, PagedKVConfig, PrefixSpan, StreamLayout
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.graph.base import GraphEdge, GraphNode, Loop, Sequential
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import ForwardPassArgs, KVLabel, LabelLayout, Model, PrefixKeys, ProcessPromptOutput, Span
from mstar.utils.ipc_format import ConductorMessage, ConductorMessageType, NewRequestConductor, SetupDone

PS = 16
PROMPT = 100
LAYOUT = {"kv": {
    "main": LabelLayout(
        spans=(Span("ids", "prefill", PROMPT, PROMPT, "text_inputs"),), growth=(("decode_loop", 1),),
    ),
    "uncond": LabelLayout(
        spans=(Span("rows", "prefill", 10, 10, "cond"),), growth=(("talk_loop", 2),), scratch=7,
    ),
}}


class _Comm:

    def __init__(self):
        self.sent = []
        self.batches = deque()

    def send(self, entity_id, msg):
        self.sent.append(msg)

    def get_all_new_messages(self):
        return self.batches.popleft() if self.batches else []


class _ScriptDoneError(Exception):
    pass


class _Model(Model):

    def __init__(self, layout=LAYOUT, max_seq_len=4096):
        self._layout, self._max_seq_len = layout, max_seq_len

    def get_node_resources(self):
        return [KVSpec(resource_key="kv", nodes={"llm"}, config=PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=self._max_seq_len, max_num_pages=64, page_size=PS,
        ))]

    def kv_labels(self):
        return {"kv": {
            "main": KVLabel(
                growth=(("decode", "decode_loop"),), keys=PrefixKeys("text_inputs", ("prefill",)),
            ),
            "uncond": KVLabel(growth=(("decode", "talk_loop"),)),
        }}

    def get_graph_walk_graphs(self):
        llm = GraphNode(name="llm", input_names=["text_inputs"], outputs=[
            GraphEdge(next_node=EMIT_TO_CLIENT, name="new_token", output_modality="text"),
        ])
        return {"prefill": llm, "decode": Sequential(sections=[
            Loop(name="decode_loop", max_iters=50, outputs=[], section=llm),
            Loop(name="talk_loop", max_iters=40, outputs=[], section=GraphNode(
                name="talker", input_names=["new_token"], outputs=[],
            )),
        ])}

    def get_max_output_tokens(self, **model_kwargs):
        return model_kwargs.get("max_output_tokens", 40)

    def get_loop_caps(self, **model_kwargs):
        return {"decode_loop": self.get_max_output_tokens(**model_kwargs)}

    def get_request_resource_configs(self, partition_fwd_args, model_kwargs=None):
        return {"kv": KVReqConfig(needed_labels=["main", "uncond"])}

    def get_initial_forward_pass_args(self, partition_name, input_modalities, output_modalities,
                                      input_signals, model_kwargs=None):
        return ForwardPassArgs(
            full_metadata=CurrentForwardConductorMetadata(graph_walk="prefill", is_prefill=True),
            inputs=[], unpersist_tensors=[],
        )

    def get_partition_forward_pass_args(self, *args, **kwargs):
        raise AssertionError("the request never leaves its first walk")

    def max_input_files(self):
        return {}

    def process_prompt(self, *args, **kwargs):
        tensors = {"text_inputs": [torch.arange(PROMPT)]}
        return tensors if self._layout is None else ProcessPromptOutput(tensors, {"kv_layout": self._layout})

    def postprocess(self, *args, **kwargs):
        return b""

    def get_submodule(self, *args, **kwargs):
        return None


def _preprocess(model: _Model, model_kwargs: dict) -> dict:
    worker = PreprocessWorkerThread(
        *(queue.Queue() for _ in range(8)), stop_event=threading.Event(), communicator=_Comm(),
        tensor_manager=_StubTensorManager(), model=model,
        model_config={"model": "stub", "resources": {"kv": {"page_size": PS}}},
    )
    worker._process_input(PreprocessInput(
        request_id="r", text="hello", file_paths=None, input_modalities=["text"],
        output_modalities=["text"], model_kwargs=model_kwargs,
    ))
    return worker.communicator.sent[-1].body.model_kwargs


def _conduct(model: _Model, model_kwargs: dict, tmp_path, monkeypatch, **config) -> _Comm:
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({
        "max_seq_len": 4096, "node_groups": [{"node_names": ["llm", "talker"], "ranks": [0]}], **config,
    }))
    comm = _Comm()
    monkeypatch.setattr(Conductor, "_launch_workers", lambda self: None)
    monkeypatch.setattr(conductor_mod, "make_communicator", lambda **kw: comm)
    conductor = comm.conductor = Conductor(model, str(path), socket_path_prefix=str(tmp_path / "s"))
    comm.batches.extend([
        [ConductorMessage(ConductorMessageType.SETUP_DONE, SetupDone(w)) for w in conductor.worker_ids],
        [ConductorMessage(ConductorMessageType.NEW_REQUEST, NewRequestConductor(
            request_id="r", initial_signals={}, initial_input_modalities=["text"],
            initial_output_modalities=["text"], input_metadata={}, model_kwargs=model_kwargs,
        ))],
    ])

    def poll():
        if not comm.batches:
            raise _ScriptDoneError

    conductor._poll_worker_liveness = poll
    with pytest.raises(_ScriptDoneError):
        conductor.run()
    return comm


def _ingest(model: _Model, model_kwargs: dict, tmp_path, monkeypatch):
    sent = _conduct(model, model_kwargs, tmp_path, monkeypatch).sent
    return next(msg.body for msg in sent if getattr(msg.message_type, "name", "") == "NEW_REQUEST")


def _manager() -> KVManager:
    return KVManager(
        cfg=PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=4096, max_num_pages=64, page_size=PS,
        ),
        name="kv", joint_comm_group=None,
        transfer_engine_info=TransferEngineInfo("h", "h", LocalTransferEngine("h")),
        device=torch.device("cpu"), dtype=torch.float32,
    )


@pytest.mark.parametrize("prefix_cache", [True, False])
def test_a_layout_reaches_its_streams_as_footprints(tmp_path, monkeypatch, prefix_cache):
    model = _Model()
    kwargs = _preprocess(model, {"max_output_tokens": 30, "prefix_cache": prefix_cache})
    new_request = _ingest(model, kwargs, tmp_path, monkeypatch)
    cfg = new_request.request_info.resource_configs["kv"]
    kv = _manager()

    kv.ingest_request("r", cfg)

    assert kv._streams["r"]["main"].footprint == -(-(PROMPT + 30) // PS)
    assert kv._ensure_label("r", "uncond").footprint == -(-(10 + 2 * 40 + 7) // PS)


def test_a_cap_past_its_loop_is_refused(tmp_path, monkeypatch):
    model = _Model()

    sent = _conduct(model, _preprocess(model, {"max_output_tokens": 51}), tmp_path, monkeypatch).sent

    failed = [msg.body for msg in sent if getattr(msg, "message_type", None) == "request_failed"]
    assert [body.status for body in failed] == [400], "a request the loop would end short was served"
    assert not any(getattr(msg.message_type, "name", "") == "NEW_REQUEST" for msg in sent)


@pytest.mark.parametrize("context, model_kwargs", [(PROMPT + 29, {"max_output_tokens": 30}), (PROMPT - 1, {})])
def test_a_request_past_the_context_is_refused(tmp_path, monkeypatch, context, model_kwargs):
    model = _Model(max_seq_len=context)

    sent = _conduct(model, _preprocess(model, model_kwargs), tmp_path, monkeypatch).sent

    failed = [msg.body for msg in sent if getattr(msg, "message_type", None) == "request_failed"]
    assert [body.status for body in failed] == [400], "a request past the model's context was served"


def test_a_span_counts_its_advance_not_its_slots_against_the_context():
    cfg = KVReqConfig(
        kv_layout={"main": StreamLayout(spans=(PrefixSpan(100, 40, "prefill"),), growth=(("decode_loop", 1),))},
        loop_caps={"decode_loop": 50},
    )

    cfg.validate(_Model(max_seq_len=90).get_node_resources()[0])
    with pytest.raises(ValueError, match="past the model's context"):
        cfg.validate(_Model(max_seq_len=89).get_node_resources()[0])


def test_a_request_that_asks_no_length_decodes_to_the_context(tmp_path, monkeypatch):
    model = _Model(max_seq_len=PROMPT + 29)

    new_request = _ingest(model, _preprocess(model, {}), tmp_path, monkeypatch)

    assert new_request.loop_caps["decode_loop"] == 29, "the default cap ran past the context"
    assert new_request.request_info.max_tokens == 29


def test_only_the_new_request_carries_the_layout(tmp_path, monkeypatch):
    model = _Model()

    comm = _conduct(model, _preprocess(model, {"max_output_tokens": 30}), tmp_path, monkeypatch)

    [new_request] = [msg.body for msg in comm.sent if getattr(msg.message_type, "name", "") == "NEW_REQUEST"]
    sent, kept = new_request.request_info.resource_configs["kv"], comm.conductor.requests["r"].resource_configs["kv"]
    assert sent.kv_layout and sent.loop_caps, "the worker's KV manager would ingest no layout"
    assert kept.kv_layout is None and kept.loop_caps is None, "the conductor's later messages would carry the layout"


def test_a_layout_the_client_sent_is_dropped(tmp_path, monkeypatch):
    model = _Model()
    kwargs = _preprocess(model, {"max_output_tokens": 30, "kv_layout": {"kv": {"main": [[[1, 1, "x"]], [], 0, None]}}})
    cfg = _ingest(model, kwargs, tmp_path, monkeypatch).request_info.resource_configs["kv"]

    assert cfg.kv_layout["main"].spans[0].length == PROMPT, "a client's layout would size another request's pages"
