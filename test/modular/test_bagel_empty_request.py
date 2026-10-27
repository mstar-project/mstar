"""A BAGEL request with neither text nor an image is refused with a 400
before the conductor sees it, on both frontends.

Without either, BAGEL's prefill schedule is empty and the request was admitted
onto a decode whose ``text_inputs`` edge carries no token. A chat whose
messages have no content reaches the model the same way. As for Orpheus,
``""`` (FastAPI's ``/generate`` reads it as missing, ``mstar-server`` passes it
through) and blank text, which were admitted and generated from BAGEL's
template alone, are refused too.

The harness is ``test_orpheus_empty_text.py``'s: the API server and its data
worker loop are real, and so is BAGEL's ``process_prompt``, with a stand-in
tokenizer.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time

import pytest
import torch
from test_model_kv_layouts import _bagel
from test_orpheus_empty_text import _post, _Server

from mstar.api_server import entrypoint

REFUSAL = "BAGEL requires a text prompt or an image"


@pytest.fixture
def server(monkeypatch, tmp_path):
    s = _Server(model=_bagel(), model_name="bagel")
    s.api.upload_dir = tmp_path
    monkeypatch.setattr(entrypoint, "api_server", s.api)
    yield s
    s.close()


TEXT_FIELDS = {"empty": {"text": ""}, "omitted": {}, "blank": {"text": " \n\t"}}


@pytest.mark.parametrize("output", ["text", "image"])
@pytest.mark.parametrize("field", TEXT_FIELDS.values(), ids=TEXT_FIELDS.keys())
def test_generate_without_text_or_an_image_is_a_400(server, field, output):
    r = _post({**field, "output_modalities": output, "streaming": "false"})

    assert r.status_code == 400, r.text
    assert REFUSAL in r.json()["detail"]
    assert server.new_requests() == [], "a request with nothing to prefill reached the conductor"


@pytest.mark.parametrize("field", TEXT_FIELDS.values(), ids=TEXT_FIELDS.keys())
def test_a_streamed_generate_without_text_or_an_image_ends_with_a_400_error_line(server, field):
    r = _post({**field, "output_modalities": "text"})

    assert r.status_code == 200, r.text
    (line,) = [json.loads(raw) for raw in r.text.splitlines()]
    assert line["modality"] == "error"
    assert line["metadata"] == {"status": 400}
    assert REFUSAL in base64.b64decode(line["data"]).decode()
    assert server.new_requests() == []


def test_a_chat_with_no_content_is_a_400(server):
    r = _post(path="/v1/chat/completions", json_body={
        "model": "bagel", "messages": [{"role": "user", "content": ""}]})

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
            "input_modalities": [], "output_modalities": ["text"],
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


@pytest.mark.parametrize("prompt", [None, ""], ids=["none", "empty"])
@pytest.mark.parametrize("output", ["text", "image"])
def test_an_image_alone_is_still_prefilled(prompt, output):
    """Understanding an image, or editing it, with no text: the image is
    the prefill, so the request is laid out rather than refused."""
    out = _bagel().process_prompt(prompt, ["image"], [output], tensors={"image_inputs": [torch.rand(3, 64, 64)]})

    assert out.metadata["kv_layout"]["kv"]["main"].spans
