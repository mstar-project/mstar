"""A request whose first sampled token is a stop token ends at the prefill (GPU).

Serves the model with ``test/e2e``'s server harness and reads the streamed
``/generate`` output. Every sampled token reaches the client as one text chunk
(a stop token's chunk is empty, the detokenizer drops it), so a request that
stops at the prefill streams exactly one text chunk. Before the fix the decode
walk started anyway and streamed more.

Needs CUDA and the weights (downloaded by the server if not cached):

    pytest test/integration/test_prefill_stop_real_weights.py -q

Each case skips when the model does not sample a stop token first, since then
it cannot exercise the path.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import pytest
import requests
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "test" / "e2e"))


def _serve(config: str, tmp_path_factory):
    from server import Server

    log = tmp_path_factory.mktemp("prefill_stop") / "server.log"
    # eager: the stop path does not depend on compiled kernels, and startup is shorter
    return Server(REPO, REPO / config, log, env={"TORCH_COMPILE_DISABLE": "1"})


def _text_chunks(url: str, model_kwargs: dict, text: str | None = None, files=None) -> list[bytes]:
    """The streamed text chunks of one ``/generate`` request, in order."""
    data = {"output_modalities": "text", "streaming": "true", "model_kwargs": json.dumps(model_kwargs)}
    if text is not None:
        data["text"] = text
    r = requests.post(url + "/generate", data=data, files=files, stream=True, timeout=600)
    r.raise_for_status()
    chunks = []
    for line in r.iter_lines():
        if not line:
            continue
        chunk = json.loads(line)
        if chunk["modality"] == "error":
            raise AssertionError(base64.b64decode(chunk["data"]).decode(errors="replace"))
        if chunk["modality"] == "text":
            chunks.append(base64.b64decode(chunk["data"]))
    return chunks


def _assert_ends_at_prefill(chunks: list[bytes]) -> None:
    if not chunks or chunks[0]:
        pytest.skip(f"the model's first token was not a stop token: {chunks[:3]!r}")
    assert len(chunks) == 1, f"decoded past a first-token stop: {chunks!r}"


@pytest.fixture(scope="module")
def qwen3_asr_url(tmp_path_factory):
    with _serve("configs/qwen3_asr.yaml", tmp_path_factory) as url:
        yield url


def test_qwen3_asr_stops_when_the_prefilled_hypothesis_is_complete(qwen3_asr_url):
    """The realtime path's case: re-decode a clip with its own finished
    transcript as the assistant prefix, so the next token is end of turn."""
    import cases  # test/e2e: a LibriSpeech utterance as WAV

    try:
        wav, _ = cases.speech_sample()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"speech sample unavailable: {exc}")
    files = {"files": ("clip.wav", wav, "audio/wav")}
    kwargs = {"language": "en", "max_output_tokens": 256}

    first = _text_chunks(qwen3_asr_url, kwargs, files=files)
    transcript = b"".join(first).decode()
    assert transcript.strip(), "the clip should transcribe to something"

    again = _text_chunks(qwen3_asr_url, {**kwargs, "assistant_prefix": transcript}, files=files)
    _assert_ends_at_prefill(again)


@pytest.fixture(scope="module")
def qwen3_5_url(tmp_path_factory):
    with _serve("configs/qwen3_5_0.8b.yaml", tmp_path_factory) as url:
        yield url


# Prompts a chat model tends to answer with an empty turn; the first one that
# does is the case.
EMPTY_REPLY_PROMPTS = (
    "Reply with an empty message: output nothing at all, not even a space or punctuation.",
    "Say nothing. Your entire reply must be empty.",
    "Do not respond. End your turn immediately without writing anything.",
)


def test_qwen3_5_stops_on_an_empty_reply(qwen3_5_url):
    kwargs = {"enable_thinking": False, "max_output_tokens": 64, "temperature": 0.0}
    for prompt in EMPTY_REPLY_PROMPTS:
        chunks = _text_chunks(qwen3_5_url, kwargs, text=prompt)
        if chunks and not chunks[0]:
            break
    _assert_ends_at_prefill(chunks)
