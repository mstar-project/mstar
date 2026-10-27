"""The request boundary of the generation contract:
null is unset, a value is coerced only when it denotes exactly the knob's type,
ranges are checked once for every endpoint, length aliases fold into
``max_output_tokens``, unread keys are logged, and a refused request gets its
status before a stream opens."""

import asyncio
import types

import pytest
from fastapi.testclient import TestClient

from mstar.api_server import entrypoint
from mstar.api_server.entrypoint import APIServer
from mstar.api_server.request_types import ResultChunk
from mstar.conductor.conductor import Conductor
from mstar.model.base import Model
from mstar.utils.generation_kwargs import check_seed, normalize_generation_kwargs


def test_null_is_unset():
    assert normalize_generation_kwargs({"temperature": None, "top_k": 5}) == {"top_k": 5}
    assert normalize_generation_kwargs(None) == {}


@pytest.mark.parametrize("kwargs", [
    {"temperature": 0}, {"temperature": 1e6}, {"top_p": 1}, {"top_p": 1e-6},
    {"min_p": 0}, {"min_p": 1}, {"repetition_penalty": 0.5}, {"top_k": 0},
    {"max_output_tokens": 1}, {"seed": -(2**63)}, {"ignore_eos": False}, {"do_sample": False},
    {"talker_temperature": 0.9}, {"code_predictor_top_k": 50},
    {"class_temperature": 0}, {"cfg_weight": "anything: not a knob"},
])
def test_in_contract_values_pass(kwargs):
    assert normalize_generation_kwargs(kwargs) == kwargs


@pytest.mark.parametrize("kwargs, name", [
    ({"temperature": -1}, "temperature"),
    ({"temperature": float("nan")}, "temperature"),
    ({"temperature": True}, "temperature"),
    ({"temperature": "nan"}, "temperature"),
    ({"temperature": ""}, "temperature"),
    ({"top_p": 0}, "top_p"),
    ({"top_p": 1.5}, "top_p"),
    ({"min_p": 2}, "min_p"),
    ({"repetition_penalty": 0}, "repetition_penalty"),
    ({"top_k": -1}, "top_k"),
    ({"top_k": 5.5}, "top_k"),
    ({"top_k": "5.5"}, "top_k"),
    ({"top_k": "5e0"}, "top_k"),
    ({"top_k": "five"}, "top_k"),
    ({"top_k": True}, "top_k"),
    ({"max_output_tokens": "0"}, "max_output_tokens"),
    ({"max_output_tokens": 0}, "max_output_tokens"),
    ({"seed": 2**64}, "seed"),
    ({"seed": 1e20}, "seed"),
    ({"ignore_eos": "false"}, "ignore_eos"),
    ({"do_sample": "false"}, "do_sample"),
    ({"num_inference_steps": 0}, "num_inference_steps"),
    ({"num_inference_steps": 7.9}, "num_inference_steps"),
    ({"height": True}, "height"),
    ({"width": "512.5"}, "width"),
    ({"num_frames": -1}, "num_frames"),
    ({"fps": 0}, "fps"),
    ({"video_fps": "-24"}, "video_fps"),
    ({"guidance_scale": "inf"}, "guidance_scale"),
    ({"talker_top_p": 0}, "talker_top_p"),
    ({"thinker_temperature": -0.1}, "thinker_temperature"),
])
def test_out_of_contract_values_raise(kwargs, name):
    with pytest.raises(ValueError, match=name):
        normalize_generation_kwargs(kwargs)


@pytest.mark.parametrize("kwargs, expected", [
    ({"temperature": "0.7"}, {"temperature": 0.7}),
    ({"talker_temperature": " 0.7 "}, {"talker_temperature": 0.7}),
    ({"top_k": 2.0}, {"top_k": 2}),
    ({"top_k": "5"}, {"top_k": 5}),
    ({"top_k": " -0 "}, {"top_k": 0}),
    ({"top_k": "5.0"}, {"top_k": 5}),
    ({"code_predictor_top_k": "50"}, {"code_predictor_top_k": 50}),
    ({"max_output_tokens": "100"}, {"max_output_tokens": 100}),
    ({"max_tokens": "64"}, {"max_output_tokens": 64}),
    ({"seed": "7"}, {"seed": 7}),
    ({"seed": 42.0}, {"seed": 42}),
    ({"seed": "9007199254740993.0"}, {"seed": 9007199254740993}),
    ({"num_inference_steps": "8", "height": 1024.0}, {"num_inference_steps": 8, "height": 1024}),
    ({"fps": "24", "guidance_scale": "-2"}, {"fps": 24.0, "guidance_scale": -2.0}),
])
def test_values_that_denote_the_knobs_type_are_coerced(kwargs, expected):
    out = normalize_generation_kwargs(kwargs)
    assert out == expected
    # the type too: 2.0 == 2, but a float top_k is what the coercion removes
    assert {k: type(v) for k, v in out.items()} == {k: type(v) for k, v in expected.items()}


def test_length_aliases_fold_into_max_output_tokens_in_order():
    assert normalize_generation_kwargs({"max_tokens": 9}) == {"max_output_tokens": 9}
    assert normalize_generation_kwargs(
        {"max_tokens": 9, "max_completion_tokens": 8, "max_new_tokens": 7}
    ) == {"max_output_tokens": 7}
    assert normalize_generation_kwargs(
        {"max_output_tokens": 5, "max_new_tokens": 7}
    ) == {"max_output_tokens": 5}
    with pytest.raises(ValueError, match="max_output_tokens"):
        normalize_generation_kwargs({"max_tokens": 0})


class _Model:
    ENGINE_KWARGS = Model.ENGINE_KWARGS
    LENGTH_KWARGS = Model.LENGTH_KWARGS

    def request_kwargs(self):
        return frozenset({"temperature"})


class _RecordingServer:
    """``APIServer`` with only the submit-side checks real."""
    submit_request = entrypoint.APIServer.submit_request
    _report_ignored_params = entrypoint.APIServer._report_ignored_params
    open_result_stream = entrypoint.APIServer.open_result_stream
    async_stream_results = entrypoint.APIServer.async_stream_results
    _stream_ndjson = entrypoint.APIServer._stream_ndjson
    _chunk_to_ndjson = entrypoint.APIServer._chunk_to_ndjson
    enable_nvtx = False

    def __init__(self, chunks=()):
        self.model = _Model()
        self.chunks = list(chunks)
        self.submitted = []

    async def wait_for_request_id(self, request_id):
        pass

    def __getattr__(self, name):
        raise AttributeError(name)


def _patch_submit(monkeypatch, server):
    def submit_request(self, **kwargs):
        kwargs["model_kwargs"] = normalize_generation_kwargs(kwargs.get("model_kwargs"))
        self._report_ignored_params(kwargs["model_kwargs"])
        self.submitted.append(kwargs)
        return "r1"

    async def iter_result_chunks(self, request_id):
        for chunk in self.chunks:
            yield chunk

    monkeypatch.setattr(_RecordingServer, "submit_request", submit_request)
    monkeypatch.setattr(_RecordingServer, "iter_result_chunks", iter_result_chunks, raising=False)
    monkeypatch.setattr(entrypoint, "api_server", server)


def test_generate_logs_keys_the_model_does_not_read(monkeypatch, caplog):
    server = _RecordingServer([ResultChunk(request_id="r1", modality="text", data=b"hi")])
    _patch_submit(monkeypatch, server)
    with caplog.at_level("WARNING", logger=entrypoint.logger.name):
        response = TestClient(entrypoint.app).post(
            "/generate",
            data={"model_kwargs": '{"temperature": 0.5, "temprature": 0.5, "seed": 3}',
                  "streaming": "true"},
        )
    assert response.status_code == 200
    # seed is the server's own; the typo is logged, the request still served
    [record] = [r for r in caplog.records if "does not read" in r.getMessage()]
    assert record.args == (["temprature"],)


def test_generate_rejects_bad_knob_with_400(monkeypatch):
    _patch_submit(monkeypatch, _RecordingServer())
    response = TestClient(entrypoint.app).post(
        "/generate", data={"model_kwargs": '{"top_p": 0}', "streaming": "false"},
    )
    assert response.status_code == 400
    assert "top_p" in response.json()["detail"]


def test_a_refused_stream_gets_its_status_not_a_200(monkeypatch):
    refused = ResultChunk(
        request_id="r1", modality="error", data=b"this model does not support min_p",
        metadata={"status": 400},
    )
    _patch_submit(monkeypatch, _RecordingServer([refused]))
    response = TestClient(entrypoint.app).post("/generate", data={"streaming": "true"})
    assert response.status_code == 400
    assert "min_p" in response.json()["detail"]


def test_an_unread_key_warns_on_every_request(caplog):
    server = _RecordingServer()
    with caplog.at_level("DEBUG", logger=entrypoint.logger.name):
        for _ in range(3):
            server._report_ignored_params({"initial_prompt": "x"})
        server._report_ignored_params({"seed": 1, "prefix_cache": True, "max_output_tokens": 4})
    levels = [r.levelname for r in caplog.records if "does not read" in r.getMessage()]
    # keys the conductor reads are never reported
    assert levels == ["WARNING"] * 3


def test_a_client_gone_before_the_first_chunk_is_aborted():
    unwound = []

    class _Server(_RecordingServer):
        async def iter_result_chunks(self, request_id):
            try:
                await asyncio.Event().wait()   # a long prefill
                yield None
            finally:
                unwound.append(request_id)   # the real iterator aborts here

    class _GoneClient:
        async def receive(self):
            return {"type": "http.disconnect"}

    async def run():
        chunks = await _Server().open_result_stream("r1", _GoneClient())
        return [c async for c in chunks]

    assert asyncio.run(asyncio.wait_for(run(), timeout=5)) == []
    assert unwound == ["r1"]


def _conductor(default=64):
    c = Conductor.__new__(Conductor)
    c.model = types.SimpleNamespace(
        get_max_output_tokens=lambda **kw: kw.get("max_output_tokens", default),
    )
    return c


@pytest.mark.parametrize("bad", ["100", 0, -5, 1.5, True])
def test_conductor_refuses_a_cap_the_run_loop_cannot_compare(bad):
    """A cap the run loop cannot compare is refused at admission; in the loop it would raise and hang."""
    with pytest.raises(ValueError, match="max_output_tokens"):
        _conductor()._max_output_tokens({"max_output_tokens": bad})


def test_rust_bridge_carries_a_refusal_as_a_400_chunk():
    """The Rust frontend turns an in-band error chunk into its status; a bare
    ``err`` frame would be a 500."""
    from mstar.api_server.rust_frontend import RustFrontendBridge

    bridge = RustFrontendBridge.__new__(RustFrontendBridge)
    sent = []
    bridge._send = sent.append
    bridge.server = types.SimpleNamespace(
        submit_request=lambda **kw: normalize_generation_kwargs(kw["model_kwargs"]),
    )
    bridge._submit({"rid": "r1", "model_kwargs": {"seed": 2**64}})
    [frame] = sent
    assert (frame["t"], frame["modality"], frame["metadata"]) == ("chunk", "error", {"status": 400})


def test_rust_bridge_passes_extras_to_the_coercing_check():
    """The Rust frontend's extra map is verbatim; ``submit_request`` coerces it."""
    from mstar.api_server.rust_frontend import RustFrontendBridge

    async def no_chunks(rid):
        return
        yield

    bridge = RustFrontendBridge.__new__(RustFrontendBridge)
    sent, seen = [], []
    bridge._keys = {}
    bridge._send = sent.append
    bridge.server = types.SimpleNamespace(
        submit_request=lambda **kw: seen.append(normalize_generation_kwargs(kw["model_kwargs"])),
        submission_key=lambda rid: rid,
        iter_result_chunks=no_chunks,
    )

    async def run():
        bridge._submit({"rid": "r1", "model_kwargs": {"top_k": "5", "talker_temperature": "0.7"}})
        await asyncio.sleep(0)

    asyncio.run(run())
    assert seen == [{"top_k": 5, "talker_temperature": 0.7}]
    assert sent == [{"t": "done", "rid": "r1", "sub": None}]


def test_checkpoint_defaults_read_both_greedy_spellings(tmp_path):
    import json

    from mstar.model.utils import load_generation_defaults

    (tmp_path / "generation_config.json").write_text(json.dumps({
        "temperature": 0.9, "top_k": 50, "max_new_tokens": 8192,
        "subtalker_dosample": False, "subtalker_top_p": 0.8,
    }))
    assert load_generation_defaults(tmp_path) == {
        "temperature": 0.9, "top_k": 50, "max_output_tokens": 8192,
    }
    assert load_generation_defaults(tmp_path, prefix="subtalker_", stage="code_predictor_") == {
        "code_predictor_top_p": 0.8, "code_predictor_temperature": 0.0,
    }
    assert load_generation_defaults(tmp_path / "missing") == {}



# ── seed ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("seed", [None, 0, 7, -(2**63), 2**63 - 1])
def test_int64_seeds_pass(seed):
    check_seed(seed)


@pytest.mark.parametrize("seed", ["abc", "7", 1.5, True, 2**63, 2**64, -(2**63) - 1, [1]])
def test_bad_seeds_raise(seed):
    with pytest.raises(ValueError, match="seed"):
        check_seed(seed)


def test_submit_request_refuses_a_bad_seed_before_registering():
    server = APIServer.__new__(APIServer)   # no state: the check must come first
    with pytest.raises(ValueError, match="64-bit"):
        server.submit_request(
            text="hi", input_modalities=["text"], output_modalities=["text"],
            model_kwargs={"seed": 2**64},
        )


def test_a_client_still_waiting_keeps_its_stream():
    """A pending first read is not mistaken for a disconnect, and nothing polls."""
    ok = ResultChunk(request_id="r1", modality="text", data=b"hi")

    class _Server(_RecordingServer):
        async def iter_result_chunks(self, request_id):
            await asyncio.sleep(0.01)
            yield ok

    class _WaitingClient:
        receives = 0

        async def receive(self):
            self.receives += 1
            await asyncio.Event().wait()

    async def run():
        client = _WaitingClient()
        chunks = await _Server().open_result_stream("r1", client)
        return [c async for c in chunks], client.receives

    assert asyncio.run(asyncio.wait_for(run(), timeout=5)) == ([ok], 1)


@pytest.mark.parametrize(("key", "sent", "read"), [
    ("num_step", 16.0, 16), ("num_step", "16", 16), ("rollout_horizon", 4.0, 4),
    ("num_inference_steps", "16.0", 16), ("speed", "1.5", 1.5),
])
def test_model_knobs_follow_the_shared_rule(key, sent, read):
    """One rule per kind of value: an integral 16.0 is 16 for every integer knob."""
    assert normalize_generation_kwargs({key: sent})[key] == read


@pytest.mark.parametrize(("key", "bad"), [
    ("num_step", 0), ("num_step", 1.5), ("rollout_horizon", -1), ("speed", 0), ("speed", "nan"),
])
def test_model_knobs_outside_their_rule_are_refused(key, bad):
    with pytest.raises(ValueError, match=key):
        normalize_generation_kwargs({key: bad})
