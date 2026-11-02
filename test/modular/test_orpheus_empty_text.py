"""An Orpheus request with no text is refused with a 400 before the conductor
sees it, on both frontends.

FastAPI reads an empty ``text`` form field as missing, so ``/generate`` hands
``None`` to ``process_prompt``; the Rust frontend passes ``""`` through. Either
way Orpheus has nothing to speak. Before this check ``process_prompt`` returned
no ``text_inputs`` for ``None``, the request was admitted, and the LLM's
``prepare_inputs`` failed with an ``IndexError`` (a 500); ``""`` and blank text
were admitted and generated from the voice prefix alone.

The API server and its data worker loop are real, and so is Orpheus's
``process_prompt``; only its tokenizer is a stand-in (the checkpoint's is not
downloaded on CPU runners). The conductor is a socket that records what it
is sent.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from types import SimpleNamespace

import httpx
import pytest
import torch
from api_server_helpers import StandInSocket, make_test_api_server
from test_request_id_reuse import _Socket, _TensorManager

from mstar.api_server import entrypoint
from mstar.api_server.data_worker import PreprocessWorker, PreprocessWorkerThread, ResultQueue
from mstar.communication.event import EventWakeup
from mstar.model.orpheus.config import OrpheusModelConfig
from mstar.model.orpheus.orpheus_model import OrpheusModel
from mstar.utils.ipc_format import ConductorMessageType

REFUSAL = "Orpheus requires a non-empty text prompt"


def _tokenize(text, return_tensors=None):
    return SimpleNamespace(input_ids=torch.tensor([[ord(c) for c in text]], dtype=torch.long))


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
        results = EventWakeup()
        worker = PreprocessWorkerThread(
            in_queue=queue.Queue(),
            result_tensor_queue=queue.Queue(),
            out_queue=ResultQueue(results),
            profile_queue=ResultQueue(results),
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
        facade.results_wakeup = results
        facade.received_chunk_counts = {}
        facade.per_request_reading_tensors = {}
        self.api = make_test_api_server(preprocess_worker=facade, model=worker.model, model_name="orpheus")
        self.api.communicator = StandInSocket()
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


def _post(form=None, path="/generate", json_body=None):
    async def run():
        transport = httpx.ASGITransport(app=entrypoint.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(path, data=form, json=json_body)

    return asyncio.run(run())


TEXT_FIELDS = {"empty": {"text": ""}, "omitted": {}, "blank": {"text": " \n\t"}}


@pytest.mark.parametrize("field", TEXT_FIELDS.values(), ids=TEXT_FIELDS.keys())
def test_generate_without_text_is_a_400(server, field):
    form = {**field, "output_modalities": "audio", "streaming": "false", "model_kwargs": '{"voice": "tara"}'}

    r = _post(form)

    assert r.status_code == 400, r.text
    assert REFUSAL in r.json()["detail"]
    assert server.new_requests() == [], "a request with no text reached the conductor"


@pytest.mark.parametrize("field", TEXT_FIELDS.values(), ids=TEXT_FIELDS.keys())
def test_a_streamed_generate_without_text_is_a_400(server, field):
    """A stream waits for its first chunk, so a preprocess refusal is the status, not a 200."""
    r = _post({**field, "output_modalities": "audio", "model_kwargs": '{"voice": "tara"}'})

    assert r.status_code == 400, r.text
    assert REFUSAL in r.json()["detail"]
    assert server.new_requests() == []


@pytest.mark.parametrize("stream", [False, True])
def test_speech_with_empty_input_is_a_400(server, stream):
    r = _post(path="/v1/audio/speech", json_body={"input": "", "voice": "tara", "stream": stream})

    assert r.status_code == 400, r.text
    assert REFUSAL in r.json()["error"]["message"]
    assert server.new_requests() == []


@pytest.mark.parametrize("text", ["", " \n\t"], ids=["empty", "blank"])
def test_rust_frontend_submit_without_text_is_a_400(server, text):
    """``mstar-server`` sends the form's ``text`` as is, so ``""`` arrives as
    ``""``, not ``None``; the bridge relays the refusal with its status."""
    from mstar.api_server.rust_frontend import RustFrontendBridge

    bridge = RustFrontendBridge.__new__(RustFrontendBridge)
    bridge.server = server.api
    bridge._keys = {}
    sent = []
    bridge._send = sent.append

    async def run():
        bridge._submit({
            "t": "submit", "rid": "X", "sub": 1, "text": text,
            "input_modalities": [], "output_modalities": ["audio"],
        })
        deadline = time.monotonic() + 10
        while not any(m["t"] in ("done", "err") for m in sent):
            assert time.monotonic() < deadline, sent
            await asyncio.sleep(0.005)

    asyncio.run(run())

    (chunk,) = [m for m in sent if m["t"] == "chunk"]
    assert chunk["modality"] == "error" and chunk["metadata"] == {"status": 400}
    assert REFUSAL in chunk["data"].decode()
    assert server.new_requests() == []


def test_text_is_still_tokenized_with_its_voice():
    (ids,) = _orpheus().process_prompt("hi", ["text"], ["audio"], voice="tara").new_input_tensors["text_inputs"]
    config = OrpheusModelConfig()
    assert ids.tolist() == [config.start_token_id, *map(ord, "tara: hi"), *config.end_token_ids]
