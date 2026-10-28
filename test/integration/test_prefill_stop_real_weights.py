"""A request whose first sampled token is a stop token ends at the prefill (GPU).

Serves the model with ``test/e2e``'s server harness and reads the streamed
``/generate`` output, where every sampled token reaches the client as one text
chunk. A request that stops at the prefill streams exactly one; before the fix
the decode walk started anyway and streamed more.

The served API reports no token ids or usage counts (``usage`` is always zero),
and nothing in a request can force a stop token, so each case first learns its
greedy first token from a probe: the same request with ``ignore_eos`` and one
output token, which samples the same token but does not stop on it.

- Qwen3.5 streams a token's raw bytes with special tokens kept, and keeps the
  stop token under ``ignore_eos``, so the probe names the token exactly.
- Qwen3-ASR's output drops every special token, so an empty probe chunk names
  only "some special token". The case assumes that after the clip's own
  finished transcript it is a stop token (``<|im_end|>`` or ``<|endoftext|>``).
  If it is another special token the run decodes on, so a wrong assumption
  fails the case rather than passing it.

A case skips only when the model's greedy first token is not a stop token,
which the API gives no deterministic way to arrange.

Needs CUDA and the weights (downloaded by the server if not cached):

    pytest test/integration/test_prefill_stop_real_weights.py -q
"""

from __future__ import annotations

import base64
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import requests
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

REPO = Path(__file__).resolve().parents[2]
E2E = REPO / "test" / "e2e"

# Qwen3.5's stop tokens (`Qwen3_5Config.stop_token_ids`) as they stream
QWEN3_5_STOP_TOKENS = (b"<|im_end|>", b"<|endoftext|>")


def _load_e2e(name: str):
    """A ``test/e2e`` helper, loaded by path under a private module name so
    its generic one (``server``, ``cases``) stays out of the session."""
    key = f"_e2e_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, E2E / f"{name}.py")
        sys.modules[key] = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(sys.modules[key])
        except BaseException:
            del sys.modules[key]
            raise
    return sys.modules[key]


def _cases():
    """``test/e2e/cases.py``, which imports ``validators`` by its bare name."""
    had = sys.modules.get("validators")
    sys.modules["validators"] = _load_e2e("validators")
    try:
        return _load_e2e("cases")
    finally:
        if had is None:
            del sys.modules["validators"]
        else:
            sys.modules["validators"] = had


def _serve(config: str, tmp_path_factory):
    log = tmp_path_factory.mktemp("prefill_stop") / "server.log"
    # eager: the stop path does not depend on compiled kernels, and startup is shorter
    return _load_e2e("server").Server(REPO, REPO / config, log, env={"TORCH_COMPILE_DISABLE": "1"})


def _text_chunks(url: str, model_kwargs: dict, text: str | None = None, wav: bytes | None = None) -> list[bytes]:
    """The streamed text chunks of one ``/generate`` request, in order."""
    data = {"output_modalities": "text", "streaming": "true", "model_kwargs": json.dumps(model_kwargs)}
    if text is not None:
        data["text"] = text
    files = {"files": ("clip.wav", wav, "audio/wav")} if wav is not None else None
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


def _first_token(url: str, model_kwargs: dict, **request) -> bytes:
    """The greedy first token's chunk, from a probe that does not stop on it."""
    chunks = _text_chunks(url, {**model_kwargs, "ignore_eos": True, "max_output_tokens": 1}, **request)
    assert len(chunks) == 1, f"one output token should stream one chunk: {chunks!r}"
    return chunks[0]


@pytest.fixture(scope="module")
def qwen3_asr_url(tmp_path_factory):
    with _serve("configs/qwen3_asr.yaml", tmp_path_factory) as url:
        yield url


def test_qwen3_asr_stops_when_the_prefilled_hypothesis_is_complete(qwen3_asr_url):
    """The realtime path's case: re-decode a clip with its own finished
    transcript as the assistant prefix, so the next token is end of turn."""
    wav = _cases().speech_wav()
    kwargs = {"language": "en", "max_output_tokens": 256}  # greedy by default

    transcript = b"".join(_text_chunks(qwen3_asr_url, kwargs, wav=wav)).decode()
    assert transcript.strip(), "the clip should transcribe to something"
    kwargs["assistant_prefix"] = transcript

    first = _first_token(qwen3_asr_url, kwargs, wav=wav)
    if first:
        pytest.skip(f"the greedy token after the transcript is text, not a stop token: {first!r}")
    chunks = _text_chunks(qwen3_asr_url, kwargs, wav=wav)
    assert chunks == [b""], (
        f"decoded past the first token: {chunks!r}. The probe's empty chunk is a "
        "special token assumed to be a stop token; check that before the stop path"
    )


@pytest.fixture(scope="module")
def qwen3_5_url(tmp_path_factory):
    with _serve("configs/qwen3_5_0.8b.yaml", tmp_path_factory) as url:
        yield url


# Prompts a chat model tends to answer with an empty turn; the first whose
# greedy first token is a stop token is the case.
EMPTY_REPLY_PROMPTS = (
    "Reply with an empty message: output nothing at all, not even a space or punctuation.",
    "Say nothing. Your entire reply must be empty.",
    "Do not respond. End your turn immediately without writing anything.",
)


def test_qwen3_5_stops_on_an_empty_reply(qwen3_5_url):
    kwargs = {"enable_thinking": False, "max_output_tokens": 64, "temperature": 0.0}
    probed = []
    for prompt in EMPTY_REPLY_PROMPTS:
        probed.append(_first_token(qwen3_5_url, kwargs, text=prompt))
        if probed[-1] in QWEN3_5_STOP_TOKENS:
            break
    else:
        pytest.skip(f"no prompt's greedy first token is a stop token: {probed!r}")
    chunks = _text_chunks(qwen3_5_url, kwargs, text=prompt)
    assert chunks == [b""], f"decoded past a first-token stop: {chunks!r}"
