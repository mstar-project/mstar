"""A chat reply's ``usage`` and ``finish_reason``.

The data worker counts the prompt's positions (``Model.prompt_tokens``) and
each text chunk's tokens, and flags the chunk a stop token ended; the
conductor returns the prompt count and the output cap it resolved on
``request_complete``, and the API server ends a ``report_usage`` submission
with a ``usage`` chunk the chat handlers turn into OpenAI's ``usage`` and
``finish_reason``. The Rust frontend's side is in ``rust/server/src/main.rs``
and ``test/rust/test_rust_frontend.py``.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import queue
import threading
from types import SimpleNamespace

import pytest
import torch
from api_server_helpers import make_test_api_server
from test_ingest_unpersist import _conductor, _info, _msg, _run, _ToyModel

from mstar.api_server.data_worker import InlineTokens, PreprocessWorkerThread, RequestOutputState
from mstar.api_server.request_types import (
    USAGE,
    APIServerMessage,
    PreprocessInput,
    RequestComplete,
    ResultChunk,
)
from mstar.api_server.served_models import Selected
from mstar.communication.event import EventWakeup
from mstar.conductor.request_info import WalkStep
from mstar.graph.base import GraphEdge, GraphNode
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import MAX_OUTPUT_TOKENS, Model, TensorAndMetadata
from mstar.model.registry import MODEL_REGISTRY, get_model_class
from mstar.model.utils import TextOutput
from mstar.utils.ipc_format import ConductorMessageType, NewRequestConductor, WorkerGraphsDone

PROMPT = "the quick brown fox jumps"


def _tokenize(text: str) -> list[int]:
    return [len(word) for word in text.split()]


# -- the data worker ----------------------------------------------------------


IMAGE_PADS = 3


class _Model:
    """Tokenizes by word and expands an image's pad in place, as the shared VLM scaffold (Qwen3.5) does."""

    def get_partition_topology(self):
        from mstar.streaming.topology import PartitionTopology
        return PartitionTopology(partitions=["default"])

    def max_input_files(self):
        return {}

    def process_prompt(self, prompt, *args, **kwargs):
        ids = torch.tensor(_tokenize(prompt), dtype=torch.long)
        pads = torch.full((IMAGE_PADS,), 99)
        return {"text_inputs": [torch.cat([ids[:2], pads, ids[2:]])], "pixel_values": [torch.rand(4, 3)]}

    prompt_tokens = Model.prompt_tokens
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
        self.stored = tensors
        return {}

    def register_for_send(self, request_id, tensor_infos, **kwargs):
        pass

    def set_persist(self, uuid, persist):
        pass


def _data_worker(model=None) -> PreprocessWorkerThread:
    return PreprocessWorkerThread(
        in_queue=queue.Queue(), result_tensor_queue=queue.Queue(), out_queue=queue.Queue(),
        profile_queue=queue.Queue(), cleanup_request_queue=queue.Queue(),
        abort_request_queue=queue.Queue(), reads_done_queue=queue.Queue(),
        discard_tensor_queue=queue.Queue(), stop_event=threading.Event(),
        communicator=_Comm(), tensor_manager=_TensorManager(), model=model or _Model(),
    )


def _sent_prompt_tokens(worker, modalities, file_paths=None) -> int:
    worker._process_input(PreprocessInput(
        request_id="r", text=PROMPT, file_paths=file_paths,
        input_modalities=modalities, output_modalities=["text"], model_kwargs={},
    ))
    [msg] = worker.communicator.sent
    assert isinstance(msg.body, NewRequestConductor)
    return msg.body.prompt_tokens


def test_an_image_prompts_pads_count_as_its_tokens():
    assert _sent_prompt_tokens(_data_worker(), ["image", "text"]) == len(_tokenize(PROMPT)) + IMAGE_PADS


def _qwen3_omni():
    pytest.importorskip("transformers")
    from test_qwen3_omni_prefix_cache import _Processor, _Tokenizer

    from mstar.model.qwen3_omni.config import Qwen3OmniModelConfig
    from mstar.model.qwen3_omni.qwen3_omni_model import Qwen3OmniModel

    model = Qwen3OmniModel.__new__(Qwen3OmniModel)
    model.tokenizer, model._processor, model.config = _Tokenizer(), _Processor(), Qwen3OmniModelConfig()
    # a 64x96 image is a 4x6 patch grid, 2x3 after the merge
    return model, 6


def _bagel():
    from test_model_kv_layouts import _bagel

    from mstar.model.bagel.submodules import vit_patches, vit_preprocess

    model = _bagel()
    cfg = model.config.vit_config.patch_size, model.config.vit_max_num_patch_per_side
    return model, vit_patches(64, 96, vit_preprocess("default"), *cfg)


@pytest.mark.parametrize("build", [_qwen3_omni, _bagel], ids=["qwen3_omni", "bagel"])
def test_a_prompt_split_around_its_image_counts_the_images_positions(build, tmp_path):
    """As OpenAI and vLLM count it: the text, the image's two sentinels and a position per image token."""
    from PIL import Image

    model, image_tokens = build()
    path = tmp_path / "image.png"
    Image.new("RGB", (96, 64)).save(path)
    worker = _data_worker(model)

    sent = _sent_prompt_tokens(worker, ["image", "text"], {"image": [str(path)]})

    # each text segment is a signal of its own (``split_spans``)
    text = sum(ids.numel() for name, segments in worker.tensor_manager.stored.items()
               if name.startswith("text_inputs") for ids in segments)
    assert sent == text + 2 + image_tokens


def test_a_whisper_request_gets_through_the_data_worker(tmp_path):
    """A real ``WhisperModel``: its forced decoder prompt is the count."""
    from test_model_kv_layouts import _whisper

    model = _whisper()
    model.load_audio = lambda filepath, device: TensorAndMetadata(
        data=torch.zeros(16000), metadata={"sample_rate": 16000, "num_channels": 1})
    worker = _data_worker(model)
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"")

    worker._process_input(PreprocessInput(
        request_id="r", text=None, file_paths={"audio": [str(clip)]},
        input_modalities=["audio"], output_modalities=["text"], model_kwargs={"language": "en"},
    ))

    [msg] = worker.communicator.sent
    assert msg.body.prompt_tokens == worker.tensor_manager.stored["text_inputs"][0].numel() > 0


@pytest.mark.parametrize("module, cls", sorted(set(MODEL_REGISTRY.values())))
def test_no_registered_model_overrides_prompt_tokens_with_other_arguments(module, cls):
    """The data worker calls ``prompt_tokens(tensors, kv_layout)`` on every
    model, so a helper of the same name with other arguments fails each request."""
    name = next(name for name, entry in MODEL_REGISTRY.items() if entry == (module, cls))
    theirs = inspect.signature(get_model_class(name).prompt_tokens)
    assert list(theirs.parameters) == list(inspect.signature(Model.prompt_tokens).parameters)


STOP_ID = 2


class _StoppingModel(_Model):
    """Declares its stop id, as the chat models' ``text_output`` does."""

    def text_output(self):
        return TextOutput(SimpleNamespace(token_bytes=lambda i: b"x", to_bytes=lambda ids: b"x" * len(ids)), {STOP_ID})


def test_a_chunk_that_ends_on_a_stop_token_says_so():
    worker = _data_worker(_StoppingModel())
    worker.request_model_kwargs["i"] = {"ignore_eos": True}
    worker._emit_inline_tokens(InlineTokens(["r", "s", "i"], [11, STOP_ID, STOP_ID], "text"))
    worker._emit_inline_tokens(InlineTokens(["r"], [STOP_ID], "audio"))
    worker.request_output_state["t"] = state = RequestOutputState()
    worker.tensor_uuid_to_metadata_per_request["t"] = {1: {}, 2: {}}
    for sequence, (uuid, ids) in enumerate([(1, [STOP_ID, 6]), (2, [5, STOP_ID])]):
        state.next_sequence += 1
        worker._postprocess_inline("t", sequence, "text", uuid, torch.tensor(ids), None)
    chunks = [worker.out_queue.get_nowait() for _ in range(6)]
    assert [(c.request_id, c.modality, c.stopped) for c in chunks] == [
        ("r", "text", False), ("s", "text", True), ("i", "text", False), ("r", "audio", False),
        ("t", "text", False), ("t", "text", True),
    ]


def test_each_text_chunk_carries_its_token_count():
    worker = _data_worker()
    worker._emit_inline_tokens(InlineTokens(["r", "s"], [11, 12], "text"))
    worker._emit_inline_tokens(InlineTokens(["r"], [13], "audio"))
    # a tensor chunk carries every id it holds; a float text tensor (Whisper's alignment) none
    worker.request_output_state["t"] = state = RequestOutputState()
    worker.tensor_uuid_to_metadata_per_request["t"] = {1: {}, 2: {}}
    for sequence, (uuid, tensor) in enumerate([(1, torch.tensor([5, 6, 7])), (2, torch.rand(2))]):
        state.next_sequence += 1
        worker._postprocess_inline("t", sequence, "text", uuid, tensor, None)
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

    c = _conductor(_ToyModel(walks, {"default": [WalkStep("prefill")]}),
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
            partition_name="default", output_signal_names=["new_token"], graph_walk="prefill",
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


def _served(token_counts, cap, report_usage=True, error=False, stopped=False):
    """A request run through ``submit_request`` and one pass of the message
    thread: text chunks carrying ``token_counts`` tokens, the last one ended by
    a stop token when ``stopped``, then its completion."""
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
    worker.chunks[-1].stopped = stopped
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


@pytest.mark.parametrize("read,token_counts,cap,stopped,completion,finish", [
    (_collected, [1, 1, 1, 1], 4, False, 4, "length"),  # max_tokens hit
    (_streamed, [1, 1, 1], 16, True, 3, "stop"),  # a stop token first
    (_streamed, [1, 3], 4, False, 4, "length"),  # a tensor chunk of several ids
    # a stop token in the last allowed slot, as OpenAI and vLLM report it
    (_collected, [1, 1, 1, 1], 4, True, 4, "stop"),
    (_streamed, [1, 3], 4, True, 4, "stop"),
], ids=["max-tokens-collected", "eos-streamed", "multi-token-chunk-streamed",
        "eos-in-the-last-slot-collected", "eos-in-the-last-slot-streamed"])
def test_a_reported_request_ends_with_its_usage(read, token_counts, cap, stopped, completion, finish):
    chunks = read(_served(token_counts, cap, stopped=stopped))
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
        return Selected(model_kwargs)

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
