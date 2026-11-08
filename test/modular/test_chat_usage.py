"""A chat reply's ``usage`` and ``finish_reason``.

The data worker counts the prompt's tokens and each text chunk's, the
conductor returns the prompt count and the output cap it resolved on
``request_complete``, and the API server ends a ``report_usage`` submission
with a ``usage`` chunk the chat handlers turn into OpenAI's ``usage`` and
``finish_reason``. The Rust frontend's side is in ``rust/server/src/main.rs``
and ``test/rust/test_rust_frontend.py``.
"""
from __future__ import annotations

import asyncio
import json
import queue
import threading
from types import SimpleNamespace

import pytest
import torch
from api_server_helpers import make_test_api_server
from test_ingest_unpersist import _conductor, _edge, _fwd, _info, _msg, _run, _ToyModel

from mstar.api_server.data_worker import InlineTokens, PreprocessWorkerThread, RequestOutputState
from mstar.api_server.request_types import (
    USAGE,
    APIServerMessage,
    PreprocessInput,
    RequestComplete,
    ResultChunk,
)
from mstar.communication.event import EventWakeup
from mstar.graph.base import GraphEdge, GraphNode
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import MAX_OUTPUT_TOKENS, Model
from mstar.utils.ipc_format import ConductorMessageType, NewRequestConductor, WorkerGraphsDone

PROMPT = "the quick brown fox jumps"


def _tokenize(text: str) -> list[int]:
    return [len(word) for word in text.split()]


# -- the data worker ----------------------------------------------------------


class _Model:
    """Tokenizes by word and splits the ids around a media span, as Qwen3.5 does."""

    def get_partition_topology(self):
        from mstar.streaming.topology import PartitionTopology
        return PartitionTopology(partitions=["default"])

    def max_input_files(self):
        return {}

    def process_prompt(self, prompt, *args, **kwargs):
        ids = torch.tensor(_tokenize(prompt), dtype=torch.long)
        return {"text_inputs": [ids[:2], ids[2:]], "pixel_values": [torch.rand(4, 3)]}

    postprocess_inline = Model.postprocess_inline
    text_output = Model.text_output

    def postprocess(self, output, modality, request_kwargs=None):
        return b"x" * output.numel()


class _Comm:
    def __init__(self):
        self.sent = []

    def get_all_new_messages(self):
        return []

    def send(self, entity, msg):
        self.sent.append(msg)


class _TensorManager:
    def store_and_return_tensor_info(self, request_id, tensors, **kwargs):
        return {}

    def register_for_send(self, request_id, tensor_infos, **kwargs):
        pass

    def set_persist(self, uuid, persist):
        pass


def _data_worker() -> PreprocessWorkerThread:
    return PreprocessWorkerThread(
        in_queue=queue.Queue(), result_tensor_queue=queue.Queue(), out_queue=queue.Queue(),
        profile_queue=queue.Queue(), cleanup_request_queue=queue.Queue(),
        abort_request_queue=queue.Queue(), reads_done_queue=queue.Queue(),
        discard_tensor_queue=queue.Queue(), stop_event=threading.Event(),
        communicator=_Comm(), tensor_manager=_TensorManager(), model=_Model(),
    )


def test_the_conductor_is_sent_the_tokenized_prompts_length():
    worker = _data_worker()
    worker._process_input(PreprocessInput(
        request_id="r", text=PROMPT, file_paths=None,
        input_modalities=["text"], output_modalities=["text"], model_kwargs={},
    ))
    [msg] = worker.communicator.sent
    assert isinstance(msg.body, NewRequestConductor)
    assert msg.body.prompt_tokens == len(_tokenize(PROMPT))


def test_each_text_chunk_carries_its_token_count():
    worker = _data_worker()
    worker._emit_inline_tokens(InlineTokens(["r", "s"], [11, 12], "text"))
    worker._emit_inline_tokens(InlineTokens(["r"], [13], "audio"))
    # a tensor chunk carries every id it holds; a float text tensor (Whisper's alignment) none
    worker.request_output_state["t"] = state = RequestOutputState()
    worker.tensor_uuid_to_metadata_per_request["t"] = {1: {}, 2: {}}
    for sequence, (uuid, tensor) in enumerate([(1, torch.tensor([5, 6, 7])), (2, torch.rand(2))]):
        state.next_sequence += 1
        worker._postprocess_inline("t", sequence, "text", uuid, tensor)
    chunks = [worker.out_queue.get_nowait() for _ in range(5)]
    assert [(c.request_id, c.modality, c.num_tokens) for c in chunks] == [
        ("r", "text", 1), ("s", "text", 1), ("r", "audio", 0), ("t", "text", 3), ("t", "text", 0),
    ]


# -- the conductor ------------------------------------------------------------


@pytest.mark.parametrize("model_kwargs,cap", [
    ({"max_output_tokens": 3}, 3),
    ({}, MAX_OUTPUT_TOKENS),
], ids=["client-cap", "model-default"])
def test_request_complete_returns_the_prompt_count_and_the_resolved_cap(tmp_path, monkeypatch, model_kwargs, cap):
    walks = {"prefill": GraphNode(name="llm", input_names=["text_inputs"], outputs=[
        GraphEdge(next_node=EMIT_TO_CLIENT, name="new_token", output_modality="text"),
    ])}

    def initial(partition, signals):
        return _fwd("prefill", [_edge("llm", "text_inputs", signals["text_inputs"])])

    c = _conductor(_ToyModel(walks, initial, lambda metadata, persist: _fwd("prefill", done=True)),
                   [{"node_names": ["llm"], "ranks": [0]}], tmp_path, monkeypatch)
    wg = c.node_walk_to_wg[("llm", "prefill")].worker_graph_id
    _run(c, [
        [_msg(ConductorMessageType.NEW_REQUEST, NewRequestConductor(
            request_id="r", initial_signals={"text_inputs": [_info(101)]},
            initial_input_modalities=["text"], initial_output_modalities=["text"],
            input_metadata={}, model_kwargs=model_kwargs, prompt_tokens=5,
        ))],
        [_msg(ConductorMessageType.WORKER_GRAPHS_DONE, WorkerGraphsDone(
            request_id="r", worker_graph_ids=[wg], is_first_tp_rank=True,
            partition_name="default", output_signal_names=["new_token"],
        ))],
    ])
    [done] = [m.body for e, m in c.comm.sent if e == "api_server" and m.message_type == "request_complete"]
    assert (done.prompt_tokens, done.max_output_tokens) == (5, cap)


# -- the API server -----------------------------------------------------------


class _DataWorker:
    """The API server's side of the data worker: hands over a fixed pass of chunks."""

    def __init__(self):
        self.inputs: list[PreprocessInput] = []
        self.chunks: list[ResultChunk] = []
        self.results_wakeup = EventWakeup()

    def new_request(self, input):
        self.inputs.append(input)

    def get_profile_updates(self):
        return []

    def get_result_chunks(self):
        out, self.chunks = self.chunks, []
        return out

    def has_pending_tensors(self, rid):
        return False

    def received_final_chunks(self, rid, counts):
        return True

    def finished_reading(self, rid, drained=True):
        pass

    def forward_abort(self, rid):
        pass


class _OnePass:
    """The API server's inbox for one message pass."""

    def __init__(self, messages):
        self.messages = messages
        self.server = None

    def get_all_new_messages(self):
        out, self.messages = self.messages, []
        return out

    def register_event_for_poll(self, event):
        pass

    def poll_for_messages(self, timeout_ms=20):
        self.server.running = False
        return False


def _served(token_counts, cap, report_usage=True, error=False):
    """A request run through ``submit_request`` and one pass of the message
    thread: text chunks carrying ``token_counts`` tokens, then its completion."""
    worker = _DataWorker()
    server = make_test_api_server(preprocess_worker=worker)
    server.submit_request(
        text=PROMPT, input_modalities=["text"], output_modalities=["text"],
        request_id="r", report_usage=report_usage,
    )
    key = worker.inputs[0].request_id
    worker.chunks = [
        ResultChunk(request_id=key, modality="text", data=b"x", num_tokens=n) for n in token_counts
    ]
    if error:
        worker.chunks.append(ResultChunk(request_id=key, modality="error", data=b"boom", metadata={"status": 500}))
    server.communicator = _OnePass([APIServerMessage(message_type="request_complete", body=RequestComplete(
        request_id=key, output_chunk_counts={"new_token": len(token_counts)},
        conductor_ingest_time=0.0, conductor_finish_time=0.0,
        prompt_tokens=len(_tokenize(PROMPT)), max_output_tokens=cap,
    ))])
    server.communicator.server = server
    server.running = True
    server._process_messages()
    return server


def _streamed(server):
    async def run():
        return [c async for batch in server.iter_result_chunk_batches("r") for c in batch]
    return asyncio.run(run())


def _collected(server):
    return asyncio.run(server.collect_results("r"))


@pytest.mark.parametrize("read,token_counts,cap,completion,finish", [
    (_collected, [1, 1, 1, 1], 4, 4, "length"),  # max_tokens hit
    (_streamed, [1, 1, 1], 16, 3, "stop"),  # a stop token first
    (_streamed, [1, 3], 4, 4, "length"),  # a tensor chunk of several ids
], ids=["max-tokens-collected", "eos-streamed", "multi-token-chunk-streamed"])
def test_a_reported_request_ends_with_its_usage(read, token_counts, cap, completion, finish):
    chunks = read(_served(token_counts, cap))
    assert [c.modality for c in chunks] == ["text"] * len(token_counts) + [USAGE]
    assert chunks[-1].request_id == "r"
    assert chunks[-1].metadata == {
        "prompt_tokens": len(_tokenize(PROMPT)), "completion_tokens": completion, "finish_reason": finish,
    }


def test_a_request_that_did_not_ask_gets_no_usage_chunk():
    assert [c.modality for c in _streamed(_served([1, 1], 2, report_usage=False))] == ["text", "text"]


def test_a_failed_request_gets_its_error_and_no_usage():
    from fastapi import HTTPException

    assert [c.modality for c in _streamed(_served([1], 4, error=True))] == ["text", "error"]
    with pytest.raises(HTTPException):
        _collected(_served([1], 4, error=True))


# -- the chat handler ---------------------------------------------------------


def _usage(finish="length"):
    return ResultChunk(request_id="r", modality=USAGE, data=b"", metadata={
        "prompt_tokens": 5, "completion_tokens": 4, "finish_reason": finish,
    })


class _ChatAPI:
    """Just enough APIServer for ``create_chat_completion``."""

    model = None

    def __init__(self, chunks, upload_dir):
        self.chunks = chunks
        self.upload_dir = upload_dir
        self.submitted = {}

    def lora_kwargs(self, model, model_kwargs):
        return model_kwargs

    def submit_request(self, **kwargs):
        self.submitted = kwargs

    async def collect_results(self, request_id, raw_request=None):
        return self.chunks

    async def open_result_stream(self, request_id, raw_request=None, batches=False):
        async def replay():
            yield self.chunks
        return replay()

    def release_request(self, request_id):
        pass


def _chat(chunks, tmp_path, **fields):
    pytest.importorskip("fastapi")
    from mstar.api_server.openai.adapters import SubmitArgs
    from mstar.api_server.openai.protocol import ChatCompletionRequest
    from mstar.api_server.openai.serving_chat import create_chat_completion

    api = _ChatAPI(chunks, tmp_path)
    adapter = SimpleNamespace(chat_to_request=lambda req, upload_dir: SubmitArgs(text="hi", input_modalities=["text"]))
    req = ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}], **fields)

    async def run():
        resp = await create_chat_completion(api, "m", adapter, req)
        if not fields.get("stream"):
            return resp
        frames = [f async for f in resp.body_iterator]
        return [json.loads(line[6:]) for f in frames for line in f.splitlines()
                if line.startswith("data: ") and line != "data: [DONE]"]

    return asyncio.run(run()), api


@pytest.mark.parametrize("finish", ["length", "stop"])
def test_a_chat_reply_reports_the_usage_chunk(finish, tmp_path):
    text = ResultChunk(request_id="r", modality="text", data=b"hello")
    resp, api = _chat([text, _usage(finish)], tmp_path)
    assert api.submitted["report_usage"] is True
    assert resp["choices"][0]["message"]["content"] == "hello"
    assert resp["choices"][0]["finish_reason"] == finish
    assert resp["usage"] == {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9}


@pytest.mark.parametrize("include_usage", [False, True])
def test_a_chat_stream_ends_with_its_finish_reason_and_usage_only_when_asked(include_usage, tmp_path):
    text = ResultChunk(request_id="r", modality="text", data=b"hello")
    events, api = _chat([text, _usage()], tmp_path, stream=True,
                        stream_options={"include_usage": include_usage})
    assert api.submitted["report_usage"] is True
    assert "".join(e["choices"][0]["delta"].get("content", "") for e in events if e["choices"]) == "hello"
    with_usage = [e for e in events if "usage" in e]
    finish = events[-2] if include_usage else events[-1]
    assert finish["choices"][0]["finish_reason"] == "length"
    if include_usage:
        assert with_usage == [events[-1]]
        assert events[-1]["choices"] == []
        assert events[-1]["usage"] == {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9}
    else:
        assert with_usage == []


@pytest.mark.parametrize("sent, reported", [("false", False), (0, False), ("yes", True), (1, True)])
def test_include_usage_reads_a_bool_as_the_rust_frontend_does(sent, reported, tmp_path):
    """Both frontends read ``include_usage`` as pydantic reads a bool (the Rust
    side's ``flex::opt_bool``): ``"false"`` asks for no usage chunk."""
    text = ResultChunk(request_id="r", modality="text", data=b"hello")
    events, _ = _chat([text, _usage()], tmp_path, stream=True, stream_options={"include_usage": sent})
    assert bool([e for e in events if "usage" in e]) is reported


def test_a_non_bool_include_usage_is_refused(tmp_path):
    pydantic = pytest.importorskip("pydantic")
    with pytest.raises(pydantic.ValidationError):
        _chat([], tmp_path, stream=True, stream_options={"include_usage": "maybe"})
