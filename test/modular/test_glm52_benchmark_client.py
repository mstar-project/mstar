"""CPU tests for the GLM-5.2 benchmark client path (ModelType.GLM52 / T2T).

No server, no GPU: OurSystem.send_request runs against a fake aiohttp
session, asserting the native /generate multipart payload shape (text,
streaming, model_kwargs JSON) and that chunk accounting over a synthetic
NDJSON stream counts tokens via the per-chunk ``token_ids`` metadata
(mstar/api_server/data_worker.py surfaces the raw ids on text chunks;
one token per chunk otherwise).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import asyncio
import base64
import json

from benchmark.base import Glm52, ModelType, RequestType, Status
from benchmark.dataset import PromptsJsonDataset
from benchmark.request import OurSystem

PROMPTS = [
    {"id": "p0", "text": "What is the capital of France?", "max_tokens": 64},
    {"id": "p1", "text": "Explain KV caching in one sentence."},
]


def _write_prompts(tmp_path: Path) -> str:
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps(PROMPTS), encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# Fake aiohttp session streaming a canned NDJSON body
# ---------------------------------------------------------------------------


class _FakeStreamContent:
    def __init__(self, lines: list[bytes]):
        self._lines = lines

    def __aiter__(self):
        async def _gen():
            for line in self._lines:
                yield line

        return _gen()


class _FakeResponse:
    def __init__(self, lines: list[bytes]):
        self.content = _FakeStreamContent(lines)

    def raise_for_status(self):
        pass


class _FakeSession:
    """Captures the multipart form passed to post() and streams canned NDJSON."""

    def __init__(self, lines: list[bytes]):
        self._lines = lines
        self.captured_url: str | None = None
        self.captured_form = None

    def post(self, url, data=None, **kwargs):
        self.captured_url = url
        self.captured_form = data
        response = _FakeResponse(self._lines)

        class _Ctx:
            async def __aenter__(self):
                return response

            async def __aexit__(self, exc_type, exc, tb):
                return False

        return _Ctx()


def _form_fields(form) -> dict[str, object]:
    """Flatten aiohttp.FormData into {field_name: value}."""
    return {opts["name"]: value for opts, _headers, value in form._fields}


def _ndjson(chunks: list[dict]) -> list[bytes]:
    return [json.dumps(c).encode() + b"\n" for c in chunks]


def _text_chunk(text: str, token_ids: list[int] | None) -> dict:
    chunk = {"modality": "text", "data": base64.b64encode(text.encode()).decode()}
    if token_ids is not None:
        chunk["metadata"] = {"token_ids": token_ids}
    return chunk


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_glm52_benchmark_model_entry():
    model = ModelType.GLM52.inst()
    assert isinstance(model, Glm52)
    # Must match mstar/model/registry.py HF_MODELS["glm52"].
    assert model.get_hf_url() == "zai-org/GLM-5.2-FP8"
    assert model.get_supported_modalities() == {RequestType.T2T}
    kwargs = model.get_model_kwargs(RequestType.T2T)
    assert kwargs["temperature"] == 0.0
    # Both spellings so the cap holds on OpenAI-shaped and native servers.
    assert kwargs["max_tokens"] == kwargs["max_output_tokens"] == 1024


def test_prompts_json_dataset_builds_t2t_requests(tmp_path):
    dataset = PromptsJsonDataset(
        filename=_write_prompts(tmp_path),
        num_requests=4,  # 2 rows resized (cycled) up to 4
    )
    requests = dataset.get_requests()
    assert len(requests) == 4
    assert all(r.req_type == RequestType.T2T for r in requests)
    assert requests[0].prompt == PROMPTS[0]["text"]
    assert requests[1].prompt == PROMPTS[1]["text"]
    assert requests[2].prompt == PROMPTS[0]["text"]
    # Per-prompt budget stamped under both spellings; absent stays absent.
    assert requests[0].model_kwargs == {"max_tokens": 64, "max_output_tokens": 64}
    assert requests[1].model_kwargs == {}


def test_glm52_native_generate_payload_and_chunk_accounting(tmp_path):
    dataset = PromptsJsonDataset(filename=_write_prompts(tmp_path), num_requests=2)
    req_input = dataset[0]
    model = ModelType.GLM52.inst()

    lines = _ndjson([
        {"modality": "text"},  # keepalive-ish chunk without data: skipped
        _text_chunk("Par", token_ids=[5432]),
        _text_chunk("is", token_ids=[98]),
        _text_chunk(" is the", token_ids=[318, 262]),  # burst: 2 ids, 1 chunk
        _text_chunk(" capital.", token_ids=None),  # no metadata: counts as 1
    ])
    session = _FakeSession(lines)

    metrics = asyncio.run(
        OurSystem().send_request(
            session=session,
            req_input=req_input,
            base_url="http://localhost:8000",
            request_id=0,
            model=model,
        )
    )

    # --- multipart payload shape --------------------------------------
    assert session.captured_url == "http://localhost:8000/generate"
    fields = _form_fields(session.captured_form)
    assert fields["text"] == PROMPTS[0]["text"]
    assert fields["streaming"] == "true"
    assert fields["output_modalities"] == "text"
    # T2T is text-in: no explicit input_modalities override, no files.
    assert "input_modalities" not in fields
    assert "files" not in fields
    model_kwargs = json.loads(fields["model_kwargs"])
    assert model_kwargs["temperature"] == 0.0
    # The prompt's 64-token budget overrides the model-level 1024 default.
    assert model_kwargs["max_tokens"] == 64
    assert model_kwargs["max_output_tokens"] == 64

    # --- chunk accounting ---------------------------------------------
    assert metrics.status == Status.SUCCESS, metrics.error
    assert metrics.response_chunks["text"] == 4  # data-less chunk skipped
    # 1 + 1 + 2 (token_ids burst) + 1 (fallback without metadata)
    assert metrics.output_text_tokens == 5
    assert "".join(metrics._text_chunks) == "Paris is the capital."
    assert "text" in metrics.ttft
    assert len(metrics.chunk_arrivals["text"]) == 4
