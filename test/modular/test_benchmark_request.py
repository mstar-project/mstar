"""benchmark/request.py and runner.py: what the client sends and how a run's figures are counted (CPU)."""

import asyncio
import base64
import json
import random
from types import SimpleNamespace

import pytest

pytest.importorskip("aiohttp")

from benchmark import asr_eval, runner  # noqa: E402
from benchmark.base import Model, ModelType, Orpheus, RequestType  # noqa: E402
from benchmark.request import (  # noqa: E402
    OursOpenAI,
    RequestInput,
    RequestMetrics,
    SGLangOmni,
    _record_vllm_omni_chunk,
    aggregate_metrics,
)
from mstar.model.qwen3_asr.constants import ASR_TEXT_TAG  # noqa: E402


class _TextModel(Model):
    def get_hf_url(self):
        return "m"

    def get_supported_modalities(self):
        return {RequestType.T2T}


class _Response:
    status = 200
    content_type = "text/event-stream"

    def __init__(self, lines):
        self._lines = lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    @property
    def content(self):
        return self

    async def _iter(self):
        for line in self._lines:
            yield line

    def __aiter__(self):
        return self._iter()

    def iter_any(self):
        return self._iter()


class _Session:
    """aiohttp.ClientSession stand-in: keeps the last payload, answers with ``lines``."""

    def __init__(self, lines=()):
        self.lines = list(lines)
        self.payload = None

    def post(self, url, json=None, **kwargs):
        self.payload = json
        return _Response(self.lines)


def _metrics(request_id, *, ok, ttft, tokens):
    m = RequestMetrics(request_id=request_id, type=RequestType.T2T)
    m.start_time = 0.0
    m.ttft = {"text": ttft}
    m.output_text_tokens = m.input_tokens = tokens
    if ok:
        m.record_completion()
        m.e2e_latency = 1.0
    else:
        m.record_error("HTTP 500")
    return m


def test_a_failed_request_counts_in_no_figure():
    """A fast failure would pull the latency stats down, and a request cut
    mid-stream would add its partial tokens to the throughput (#1043)."""
    agg = aggregate_metrics(
        [_metrics(0, ok=True, ttft=0.5, tokens=10), _metrics(1, ok=False, ttft=0.01, tokens=4)], wall_time=1.0,
    )
    assert (agg.ttft["text"].mean, agg.e2e_latency.mean) == (0.5, 1.0)
    assert (agg.total_text_tokens, agg.total_input_tokens, agg.text_token_throughput) == (10, 10, 10.0)


def test_an_audio_chunk_with_completion_tokens_zero_keeps_the_text_count():
    """vllm-omni and sglang-omni send completion_tokens=0 on audio chunks (#1045)."""
    m = RequestMetrics(request_id=0, type=RequestType.T2S)
    _record_vllm_omni_chunk({"usage": {"completion_tokens": 7}, "choices": []}, m)
    _record_vllm_omni_chunk({"usage": {"completion_tokens": 0}, "modality": "audio", "choices": []}, m)
    assert m.output_text_tokens == 7

    lines = [b"data: " + json.dumps({"usage": {"completion_tokens": n}, "choices": []}).encode() for n in (7, 0)]
    req = RequestInput(req_type=RequestType.T2T, prompt="hi")
    m = asyncio.run(SGLangOmni().send_request(_Session(lines + [b"data: [DONE]"]), req, "http://x", 0, _TextModel()))
    assert m.output_text_tokens == 7


@pytest.mark.parametrize(
    ("system", "model", "req_type"),
    [
        pytest.param(SGLangOmni, _TextModel, RequestType.T2T, id="sglang_omni"),
        pytest.param(OursOpenAI, Orpheus, RequestType.T2S, id="mstar_audio_speech"),
    ],
)
def test_the_stamped_output_length_reaches_the_payload(system, model, req_type):
    """The runner stamps max_tokens / ignore_eos per request; an engine that
    drops them generates its own lengths in a fixed-length comparison (#1044)."""
    session = _Session()
    req = RequestInput(req_type=req_type, prompt="hi", model_kwargs={"max_tokens": 33, "ignore_eos": True})
    asyncio.run(system().send_request(session, req, "http://x", 0, model()))
    assert (session.payload["max_tokens"], session.payload["ignore_eos"]) == (33, True)


def test_an_offline_run_stamps_each_request_its_own_length_and_fills_its_last_wave(monkeypatch):
    """A dataset resized past its length repeats the same RequestInput objects:
    stamping them in place gave repeats one shared length. The offline round-up
    to a whole wave must happen before the dataset is sized."""
    cfg = runner.BenchmarkConfig(
        url="http://x", model=_TextModel(), dataset=runner.DatasetType.TEXT, num_requests=5,
        request_type=RequestType.T2T, local_cache_dir=".", num_warmup=0, batch_size=4,
        profiling_type=runner.ProfilingType.OFFLINE, output_len_min=1, output_len_max=1000,
    )
    bench = runner.Benchmark.__new__(runner.Benchmark)
    bench.config = cfg
    base = [RequestInput(req_type=RequestType.T2T, prompt=str(i)) for i in range(2)]
    bench._get_dataset = lambda: SimpleNamespace(get_requests=lambda: (base * 8)[: cfg.num_requests])
    sent = []

    async def offline(session, requests):
        sent.extend(requests)
        return []

    bench._run_concurrent_offline = offline
    monkeypatch.setattr(runner, "aggregate_metrics", lambda *a, **k: None)
    asyncio.run(bench.run())
    rng = random.Random(cfg.output_len_seed)
    assert [r.model_kwargs["max_tokens"] for r in sent] == [rng.randint(1, 1000) for _ in range(8)]


def test_qwen3_asr_is_scored_without_its_language_prefix():
    """/generate returns "language English<asr_text>..."; the prefix is not transcript (#1027)."""
    text = base64.b64encode(f"language English{ASR_TEXT_TAG}hi there".encode()).decode()
    session = _Session([json.dumps({"modality": "text", "data": text}).encode()])
    req = RequestInput(req_type=RequestType.A2T, prompt="")
    assert asyncio.run(asr_eval._transcribe(session, req, "http://x", 0, _TextModel())) == (0, "hi there", None)


def test_ministral_benchmark_sends_greedy_text_output_and_length_controls():
    model = ModelType.MINISTRAL3.inst()
    assert model.get_supported_modalities() == {RequestType.T2T, RequestType.I2T}
    session = _Session()
    req = RequestInput(req_type=RequestType.T2T, prompt="hi", model_kwargs={"max_tokens": 128, "ignore_eos": True})
    asyncio.run(OursOpenAI().send_request(session, req, "http://x", 0, model))
    assert session.payload["model"] == "mistralai/Ministral-3-3B-Instruct-2512"
    assert session.payload["modalities"] == ["text"]
    assert session.payload["messages"][0] == model.get_openai_system_message()
    assert session.payload["temperature"] == 0.0
    assert session.payload["top_p"] == session.payload["repetition_penalty"] == 1.0
    assert session.payload["max_tokens"] == 128 and session.payload["ignore_eos"]


@pytest.mark.parametrize("usage", [None, 7])
def test_openai_token_counts_distinguish_usage_from_chunks(usage):
    chunks = [{"choices": [{"delta": {"content": "hello"}}]}]
    if usage is not None:
        chunks.append({"usage": {"completion_tokens": usage}, "choices": []})
    session = _Session([b"data: " + json.dumps(chunk).encode() + b"\n\n" for chunk in chunks] + [b"data: [DONE]\n\n"])
    req = RequestInput(req_type=RequestType.T2T, prompt="hi")
    m = asyncio.run(OursOpenAI().send_request(session, req, "http://x", 0, _TextModel()))
    assert m.server_completion_tokens == usage
    assert m.output_text_tokens == (usage or 1)


@pytest.mark.parametrize("system", list(runner.InferenceSystemType))
def test_results_identify_the_measured_system(tmp_path, system):
    cfg = runner.BenchmarkConfig(
        url="http://x", model=_TextModel(), dataset=runner.DatasetType.TEXT, num_requests=1,
        request_type=RequestType.T2T, local_cache_dir=".", inference_system=system, output_dir=str(tmp_path),
    )
    bench = runner.Benchmark(cfg)
    metrics = [_metrics(0, ok=True, ttft=0.1, tokens=4)]
    bench._write_results_json(metrics, aggregate_metrics(metrics, wall_time=2), 2)
    result = json.loads((tmp_path / "results.json").read_text())
    assert result["system"] == system.value
    assert result["completed"] == 1 and result["failed"] == 0


def test_ministral_manifest_harness_preserves_failures_and_rejects_invalid_measurements(tmp_path, monkeypatch):
    from benchmark.ministral3.run import run

    manifest = tmp_path / "requests.jsonl"
    manifest.write_text('{"prompt": "first"}\n{"prompt": "second"}\n')
    output = tmp_path / "cell"
    sent = []

    async def send(self, session, req_input, base_url, request_id, model, **kwargs):
        sent.append((req_input.prompt, req_input.model_kwargs))
        m = _metrics(request_id, ok=request_id == 0, ttft=0.1, tokens=4)
        m.chunk_arrivals = {"text": [0.1, 0.2]}
        return m

    monkeypatch.setattr(OursOpenAI, "send_request", send)
    monkeypatch.setattr("benchmark.base.Ministral3.get_tokenizer",
                        lambda self: SimpleNamespace(encode=lambda *a, **k: [0] * 4))
    args = SimpleNamespace(manifest=manifest, num_requests=None, concurrency=2, output_dir=output,
                           engine="vllm", url="http://x", output_tokens=4, warmup=0)
    with pytest.raises(RuntimeError, match="1 failures"):
        asyncio.run(run(args))
    assert [prompt for prompt, _ in sent] == ["first", "second"]
    assert all(kwargs["max_tokens"] == 4 and kwargs["ignore_eos"] for _, kwargs in sent)
    raw = json.loads((output / "raw-requests.json").read_text())
    assert len(raw) == 2 and raw[1]["error"] == "HTTP 500"
    assert raw[0]["chunk_arrivals_s"]["text"] == [0.1, 0.2]
    result = json.loads((output / "results.json").read_text())
    assert result["engine"] == "vllm" and result["failed"] == 1


def test_ministral_length_check_uses_complete_text_when_usage_is_absent(tmp_path, monkeypatch):
    from benchmark.ministral3.run import run

    manifest = tmp_path / "requests.jsonl"
    manifest.write_text('{"prompt": "formula"}\n')
    output = tmp_path / "cell"

    async def send(self, session, req_input, base_url, request_id, model, **kwargs):
        m = _metrics(request_id, ok=True, ttft=0.1, tokens=1)
        m._text_chunks = ["₂".encode()]
        return m

    def encode(text, **kwargs):
        assert text == "₂"
        return [100, 101]  # Two byte tokens delivered in a single UTF-8 chunk.

    monkeypatch.setattr(OursOpenAI, "send_request", send)
    monkeypatch.setattr("benchmark.base.Ministral3.get_tokenizer",
                        lambda self: SimpleNamespace(encode=encode))
    args = SimpleNamespace(manifest=manifest, num_requests=None, concurrency=1, output_dir=output,
                           engine="mstar", url="http://x", output_tokens=2, warmup=0)
    asyncio.run(run(args))
    raw = json.loads((output / "raw-requests.json").read_text())[0]
    assert raw["output_tokens"] == 1 and raw["decoded_text_tokens"] == 2
    assert raw["output_count_source"] == "chunk_estimate"
    assert not json.loads((output / "results.json").read_text())["token_counts_verified_by_server"]
