"""An Orpheus prompt that cannot fit the model's context is refused with a 400
before the conductor sees it, naming its length and the limit.

The LLM writes the prompt and then up to ``get_max_output_tokens()`` decoded
tokens into one 131,072-position context. Nothing compared the two, so a
200,005-token prompt was admitted and prefilled at positions past the context.

The API server and its data worker loop are real, and so is Orpheus's
``process_prompt``; its tokenizer is a stand-in that makes one token per word
(the checkpoint's is not downloaded on CPU runners). The conductor is a socket
that records what it is sent.
"""

from __future__ import annotations

import asyncio
import base64
import json
import queue
import threading
import time
from types import SimpleNamespace

import httpx
import pytest
import torch
from api_server_helpers import make_test_api_server
from test_request_id_reuse import _Socket, _TensorManager

from mstar.api_server import entrypoint
from mstar.api_server.data_worker import PreprocessWorker, PreprocessWorkerThread
from mstar.model.orpheus.config import OrpheusModelConfig
from mstar.model.orpheus.orpheus_model import OrpheusModel
from mstar.utils.ipc_format import ConductorMessageType

CONFIG = OrpheusModelConfig()
# 131,072 positions less the 2,048 the decode loop may add
LIMIT = CONFIG.max_position_embeddings - 2048
# the start token, the "tara:" word and the four end tokens around the text
WRAP = 1 + 1 + len(CONFIG.end_token_ids)


def _tokenize(text, return_tensors=None):
    return SimpleNamespace(input_ids=torch.ones((1, len(text.split())), dtype=torch.long))


def _words(prompt_tokens: int) -> str:
    """Text whose prompt, wrapped and voiced, is ``prompt_tokens`` long."""
    return "a " * (prompt_tokens - WRAP)


def _orpheus() -> OrpheusModel:
    model = object.__new__(OrpheusModel)
    model.config = OrpheusModelConfig()
    model.tokenizer = _tokenize
    return model


class _Server:
    """The API server, its message thread and the data worker loop, wired to
    a recording conductor socket."""

    def __init__(self):
        self.socket = _Socket()
        self.stop = threading.Event()
        worker = PreprocessWorkerThread(
            in_queue=queue.Queue(),
            result_tensor_queue=queue.Queue(),
            out_queue=queue.Queue(),
            profile_queue=queue.Queue(),
            cleanup_request_queue=queue.Queue(),
            abort_request_queue=queue.Queue(),
            reads_done_queue=queue.Queue(),
            discard_tensor_queue=queue.Queue(),
            stop_event=self.stop,
            communicator=self.socket,
            tensor_manager=_TensorManager(),
            model=_orpheus(),
        )
        facade = PreprocessWorker.__new__(PreprocessWorker)
        facade.request_input_queue = worker.in_queue
        facade.result_tensor_input_queue = worker.result_tensor_queue
        facade.abort_request_queue = worker.abort_request_queue
        facade.reads_done_queue = worker.reads_done_queue
        facade.cleanup_request_queue = worker.cleanup_request_queue
        facade.discard_tensor_queue = worker.discard_tensor_queue
        facade.output_queue = worker.out_queue
        facade.profile_queue = worker.profile_queue
        facade.received_chunk_counts = {}
        facade.per_request_reading_tensors = {}
        self.api = make_test_api_server(preprocess_worker=facade, model=worker.model, model_name="orpheus")
        self.api.communicator = SimpleNamespace(get_all_new_messages=lambda: [])
        self.api.conductor_proc = None
        self.api._liveness_interval_s = 0.5
        self.api.running = True
        self.threads = [
            threading.Thread(target=worker.run, daemon=True),
            threading.Thread(target=self.api._process_messages, daemon=True),
        ]
        for thread in self.threads:
            thread.start()

    def new_requests(self):
        return self.socket.sent_of(ConductorMessageType.NEW_REQUEST)

    def close(self):
        self.api.running = False
        self.stop.set()
        for thread in self.threads:
            thread.join(timeout=10)


@pytest.fixture
def server(monkeypatch, tmp_path):
    s = _Server()
    s.api.upload_dir = tmp_path
    monkeypatch.setattr(entrypoint, "api_server", s.api)
    yield s
    s.close()


def _post(form):
    async def run():
        transport = httpx.ASGITransport(app=entrypoint.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post("/generate", data=form)

    return asyncio.run(run())


def _refusal(tokens: int) -> str:
    return f"The prompt is {tokens} tokens; Orpheus takes at most {LIMIT}"


@pytest.mark.parametrize("tokens", [LIMIT + 1, 200_005 + WRAP], ids=["one-over", "served-case"])
def test_generate_with_a_prompt_past_the_context_is_a_400(server, tokens):
    r = _post({"text": _words(tokens), "output_modalities": "audio", "streaming": "false"})

    assert r.status_code == 400, r.text
    assert _refusal(tokens) in r.json()["detail"]
    assert server.new_requests() == [], "a prompt past the context reached the conductor"


def test_a_streamed_generate_past_the_context_ends_with_a_400_error_line(server):
    """A stream's status is committed before the data worker runs, so the
    refusal is its one line, carrying the 400 as every preprocess refusal does."""
    r = _post({"text": _words(LIMIT + 1), "output_modalities": "audio"})

    assert r.status_code == 200, r.text
    (line,) = [json.loads(raw) for raw in r.text.splitlines()]
    assert line["modality"] == "error" and line["metadata"] == {"status": 400}
    assert _refusal(LIMIT + 1) in base64.b64decode(line["data"]).decode()
    assert server.new_requests() == []


def test_a_prompt_at_the_limit_is_admitted(server):
    server.api.submit_request(
        text=_words(LIMIT), input_modalities=["text"], output_modalities=["audio"], streaming=False,
    )
    deadline = time.monotonic() + 10
    while not server.new_requests():
        assert time.monotonic() < deadline, "the prompt at the limit never reached the conductor"
        time.sleep(0.005)
    (key,) = server.api.pending_requests
    assert server.api.pending_requests[key].error is None


def test_rust_frontend_submit_past_the_context_is_a_400(server):
    """``mstar-server`` reaches the same data worker through the bridge, which
    relays the refusal with its status."""
    from mstar.api_server.rust_frontend import RustFrontendBridge

    bridge = RustFrontendBridge.__new__(RustFrontendBridge)
    bridge.server = server.api
    bridge._keys = {}
    sent = []
    bridge._send = sent.append

    async def run():
        bridge._submit({
            "t": "submit", "rid": "X", "sub": 1, "text": _words(LIMIT + 1),
            "input_modalities": ["text"], "output_modalities": ["audio"],
        })
        deadline = time.monotonic() + 10
        while not any(m["t"] in ("done", "err") for m in sent):
            assert time.monotonic() < deadline, sent
            await asyncio.sleep(0.005)

    asyncio.run(run())

    (chunk,) = [m for m in sent if m["t"] == "chunk"]
    assert chunk["modality"] == "error" and chunk["metadata"] == {"status": 400}
    assert _refusal(LIMIT + 1) in chunk["data"].decode()
    assert server.new_requests() == []
