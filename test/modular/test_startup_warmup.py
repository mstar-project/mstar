"""``warmup_requests`` in the deployment yaml: run through the API server
after the workers are ready and before the server binds, as plain /generate
requests; a failing one is logged and skipped."""

from __future__ import annotations

import logging
import sys

sys.path.insert(0, ".")

from mstar.api_server import entrypoint


class _FakeServer:
    def __init__(self, fail_on: str | None = None):
        self.submitted = []
        self.collected = []
        self.fail_on = fail_on

    def submit_request(self, **kwargs):
        self.submitted.append(kwargs)
        return kwargs["request_id"]

    async def collect_results(self, request_id, raw_request=None):
        self.collected.append(request_id)
        if self.fail_on == request_id:
            raise RuntimeError("compile blew up")
        return []


def test_warmup_requests_run_as_generate_requests(caplog) -> None:
    fake = _FakeServer()
    specs = [
        {"text": "a drone over a coast", "output_modalities": ["video"], "model_kwargs": {"num_frames": 121}},
        {"text": "hi", "output_modalities": "text,audio"},
        {"image": "/tmp/obs.jpg", "text": "pick up the mug", "output_modalities": ["action"]},
    ]
    with caplog.at_level(logging.INFO, logger="mstar.api_server.entrypoint"):
        entrypoint._run_warmup_requests(fake, specs)
    assert [s["request_id"] for s in fake.submitted] == ["warmup-0", "warmup-1", "warmup-2"]
    assert fake.collected == ["warmup-0", "warmup-1", "warmup-2"]
    first, second, third = fake.submitted
    assert first["output_modalities"] == ["video"] and first["model_kwargs"] == {"num_frames": 121}
    assert first["streaming"] is False and first["input_modalities"] == ["text"]
    assert second["output_modalities"] == ["text", "audio"] and second["model_kwargs"] is None
    assert third["file_paths"] == {"image": ["/tmp/obs.jpg"]}
    assert third["input_modalities"] == ["image", "text"]
    assert [p.modality for p in third["prompt_parts"]] == ["image", "text"]
    assert caplog.text.count("done in") == 3


def test_a_failing_warmup_is_logged_and_skipped(caplog) -> None:
    fake = _FakeServer(fail_on="warmup-0")
    with caplog.at_level(logging.WARNING, logger="mstar.api_server.entrypoint"):
        entrypoint._run_warmup_requests(fake, [{"text": "x"}, "not a mapping", {"text": "y"}])
    assert fake.collected == ["warmup-0", "warmup-2"]
    assert "warmup request 1/3 failed" in caplog.text and "compile blew up" in caplog.text
    assert "not a mapping" in caplog.text
    # not a list at all: nothing runs, one warning
    entrypoint._run_warmup_requests(fake, {"text": "x"})
    assert fake.collected == ["warmup-0", "warmup-2"]
