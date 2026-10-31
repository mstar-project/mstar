"""API-server process settings from the deployment yaml. ``warmup_requests``
run through the API server after the workers are ready and before the server
binds, as plain /generate requests; a failing one is logged and skipped.
``api_switch_interval_s`` must be a positive number of seconds."""

from __future__ import annotations

import logging
import sys

import pytest

sys.path.insert(0, ".")

from mstar.api_server import entrypoint
from mstar.graph.base import GraphEdge, GraphNode, Sequential


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
        {"files": ["/tmp/obs.jpg"], "text": "pick up the mug", "output_modalities": ["action"]},
    ]
    with caplog.at_level(logging.INFO, logger="mstar.api_server.entrypoint"):
        entrypoint._run_warmup_requests(fake, specs)
    assert [s["request_id"] for s in fake.submitted] == ["warmup-0.0.0", "warmup-1.0.0", "warmup-2.0.0"]
    assert fake.collected == ["warmup-0.0.0", "warmup-1.0.0", "warmup-2.0.0"]
    first, second, third = fake.submitted
    assert first["output_modalities"] == ["video"] and first["model_kwargs"] == {"num_frames": 121}
    assert first["streaming"] is False and first["input_modalities"] == ["text"]
    assert second["output_modalities"] == ["text", "audio"] and second["model_kwargs"] is None
    assert third["file_paths"] == {"image": ["/tmp/obs.jpg"]}
    assert third["input_modalities"] == ["image", "text"]
    assert [p.modality for p in third["prompt_parts"]] == ["image", "text"]
    assert caplog.text.count("done in") == 3


def test_warmup_files_of_any_modality_keep_prompt_order() -> None:
    fake = _FakeServer()
    spec = {
        "files": [
            "/tmp/a.jpg",
            "/tmp/speech.wav",
            "/tmp/b.png",
            {"path": "/tmp/clip.bin", "modality": "video"},
        ],
        "text": "describe",
    }
    entrypoint._run_warmup_requests(fake, [spec])
    (req,) = fake.submitted
    assert req["file_paths"] == {
        "image": ["/tmp/a.jpg", "/tmp/b.png"],
        "audio": ["/tmp/speech.wav"],
        "video": ["/tmp/clip.bin"],
    }
    assert req["input_modalities"] == ["image", "audio", "image", "video", "text"]
    assert [(p.modality, p.index) for p in req["prompt_parts"]] == [
        ("image", 0), ("audio", 0), ("image", 1), ("video", 0), ("text", 0),
    ]
    assert req["output_modalities"] == ["text"]


def test_malformed_warmups_are_skipped(caplog) -> None:
    fake = _FakeServer()
    with caplog.at_level(logging.WARNING, logger="mstar.api_server.entrypoint"):
        entrypoint._run_warmup_requests(fake, [
            {"files": ["/tmp/blob.xyz"]},           # modality unknown
            {"image": "/tmp/a.jpg"},                # unknown key
            {"text": "x", "model_kwargs": [1]},     # model_kwargs not a mapping
            {"text": "ok"},
        ])
    assert fake.collected == ["warmup-3.0.0"]
    assert "set 'modality'" in caplog.text
    assert "unknown key(s) ['image']" in caplog.text
    assert "model_kwargs must be a mapping" in caplog.text


def test_repeat_sends_a_long_text(caplog) -> None:
    fake = _FakeServer()
    with caplog.at_level(logging.WARNING, logger="mstar.api_server.entrypoint"):
        entrypoint._run_warmup_requests(fake, [
            {"text": "ab ", "repeat": 3, "model_kwargs": {"max_output_tokens": 2}},
            {"text": "x", "repeat": 0},
            {"text": "x", "repeat": "3"},
            {"text": "x", "repeat": True},
            {"files": ["/tmp/a.jpg"], "repeat": 2},
            {"text": 5, "repeat": 3},
            # past what a str can hold: skipped, not a failed boot
            {"text": "x", "repeat": 10**19},
        ])
    assert fake.collected == ["warmup-0.0.0"]
    assert fake.submitted[0]["text"] == "ab ab ab "
    assert fake.submitted[0]["prompt_parts"][0].text == "ab ab ab "
    assert caplog.text.count("repeat must be a positive integer") == 3
    assert "repeat needs text" in caplog.text
    assert "text must be a string" in caplog.text
    assert "warmup_requests[6] skipped" in caplog.text


def test_a_failing_warmup_is_logged_and_skipped(caplog) -> None:
    fake = _FakeServer(fail_on="warmup-0.0.0")
    with caplog.at_level(logging.WARNING, logger="mstar.api_server.entrypoint"):
        entrypoint._run_warmup_requests(fake, [{"text": "x"}, "not a mapping", {"text": "y"}])
    assert fake.collected == ["warmup-0.0.0", "warmup-2.0.0"]
    assert "warmup request 1/3 failed" in caplog.text and "compile blew up" in caplog.text
    assert "not a mapping" in caplog.text
    # not a list at all: nothing runs, one warning
    entrypoint._run_warmup_requests(fake, {"text": "x"})
    assert fake.collected == ["warmup-0.0.0", "warmup-2.0.0"]
    # a failing solo does not keep the copies from running
    fake = _FakeServer(fail_on="warmup-0.solo")
    entrypoint._run_warmup_requests(fake, [{"text": "x"}], entrypoint.DataParallelReplicas(2, 1))
    assert fake.collected == ["warmup-0.solo", "warmup-0.0.0", "warmup-0.0.1"]


def test_hf_file_references_resolve_from_the_cache(monkeypatch) -> None:
    calls = []

    def fake_download(repo_id, filename):
        calls.append((repo_id, filename))
        return f"/cache/{repo_id}/{filename}"

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    assert entrypoint._resolve_warmup_path("/tmp/obs.jpg") == "/tmp/obs.jpg"
    assert (
        entrypoint._resolve_warmup_path("hf://nvidia/Cosmos3-Edge/assets/example_i2v_input.jpg")
        == "/cache/nvidia/Cosmos3-Edge/assets/example_i2v_input.jpg"
    )
    assert calls == [("nvidia/Cosmos3-Edge", "assets/example_i2v_input.jpg")]
    fake = _FakeServer()
    spec = {"files": ["hf://nvidia/Cosmos3-Edge/assets/x.jpg"], "output_modalities": ["video"]}
    entrypoint._run_warmup_requests(fake, [spec])
    assert fake.submitted[0]["file_paths"] == {"image": ["/cache/nvidia/Cosmos3-Edge/assets/x.jpg"]}
    # a malformed reference is a failed warmup, not a crash
    entrypoint._run_warmup_requests(fake, [{"files": ["hf://nvidia.jpg"], "output_modalities": ["video"]}])
    assert len(fake.submitted) == 1


def test_warmup_reaches_every_data_parallel_replica(tmp_path) -> None:
    """One concurrent copy of each warmup per replica, so the conductor's
    least-loaded placement warms every replica, not a random one. Replicas are
    counted from the model's worker graphs, the ones the conductor picks from:
    the most and the fewest any of them is served by."""
    from test_startup_graph_validation import _config, _llm, _ToyModel

    def walks():
        encoder = GraphNode(name="text_encoder", input_names=["x"], outputs=[GraphEdge(next_node="llm", name="x")])
        return {"gen": Sequential([encoder, _llm()])}

    model = _ToyModel(walks, tp_enabled_nodes={"llm"})

    def replicas(node_groups):
        return entrypoint._data_parallel_replicas(model, _config(tmp_path, node_groups))

    assert replicas([
        {"node_names": ["llm"], "ranks": [0, 1]},
        {"node_names": ["text_encoder"], "ranks": [0]},
    ]) == (2, 1)
    assert replicas([
        {"node_names": ["llm"], "ranks": [0, 1]},
        {"node_names": ["text_encoder"], "ranks": [0, 1]},
    ]) == (2, 2)
    # a lockstep instance is one replica
    assert replicas([
        {"node_names": ["llm"], "ranks": [0, 1], "tp_size": 2},
        {"node_names": ["text_encoder"], "ranks": [0]},
    ]) == (1, 1)
    assert replicas([
        {"node_names": ["llm"], "ranks": [0, 1, 2, 3], "tp_size": 2},
        {"node_names": ["text_encoder"], "ranks": [0]},
    ]) == (2, 1)
    # a node group no walk places is never routed to, so it adds no copies
    assert replicas([
        {"node_names": ["llm", "text_encoder"], "ranks": [0]},
        {"node_names": ["unused"], "ranks": [1, 2, 3]},
    ]) == (1, 1)
    # worker graphs that can't be built leave warmup at one copy, not startup stopped
    assert entrypoint._data_parallel_replicas(model, str(tmp_path / "missing.yaml")) == (1, 1)
    fake = _FakeServer()
    entrypoint._run_warmup_requests(fake, [{"text": "a"}, {"text": "b"}], entrypoint.DataParallelReplicas(2, 2))
    assert [s["request_id"] for s in fake.submitted] == ["warmup-0.0.0", "warmup-0.0.1", "warmup-1.0.0", "warmup-1.0.1"]


def test_replicated_warmup_runs_one_request_alone_first() -> None:
    """When replicas are uneven, each entry first runs one request to its end
    alone: a worker graph with fewer replicas than the copies (one DiT beside a
    replicated VAE) would otherwise get every copy together and only compile
    the batched shape. Evenly replicated, every graph already sees one copy."""
    events = []

    class _OrderedServer(_FakeServer):
        def submit_request(self, **kwargs):
            events.append(("submit", kwargs["request_id"]))
            return super().submit_request(**kwargs)

        async def collect_results(self, request_id, raw_request=None):
            events.append(("done", request_id))
            return await super().collect_results(request_id, raw_request)

    specs = [{"text": "a", "concurrency": [1, 2]}, {"text": "b"}]
    entrypoint._run_warmup_requests(_OrderedServer(), specs, entrypoint.DataParallelReplicas(2, 1))
    assert events[:2] == [("submit", "warmup-0.solo"), ("done", "warmup-0.solo")]
    assert [rid for kind, rid in events if kind == "submit"] == [
        "warmup-0.solo", "warmup-0.0.0", "warmup-0.0.1", *(f"warmup-0.1.{c}" for c in range(4)),
        "warmup-1.solo", "warmup-1.0.0", "warmup-1.0.1",
    ]
    solo_1 = events.index(("submit", "warmup-1.solo"))
    assert events[solo_1 + 1] == ("done", "warmup-1.solo")
    # every graph has as many replicas as copies: nothing extra
    events.clear()
    entrypoint._run_warmup_requests(_OrderedServer(), specs, entrypoint.DataParallelReplicas(2, 2))
    assert not any(rid.endswith(".solo") for _, rid in events)


def test_concurrency_runs_copies_together_for_batched_shapes() -> None:
    """``concurrency: N`` submits N copies at once (times the replica count), so
    a step compiled per batch size sees that batch before the bind."""
    fake = _FakeServer()
    entrypoint._run_warmup_requests(fake, [{"text": "a", "concurrency": 3}], entrypoint.DataParallelReplicas(2, 2))
    assert [s["request_id"] for s in fake.submitted] == [f"warmup-0.0.{c}" for c in range(6)]
    # a list runs one warmup per batch size, in order, each under its own ids
    fake = _FakeServer()
    entrypoint._run_warmup_requests(fake, [{"text": "a", "concurrency": [1, 2]}])
    assert [s["request_id"] for s in fake.submitted] == ["warmup-0.0.0", "warmup-0.1.0", "warmup-0.1.1"]
    fake = _FakeServer()
    entrypoint._run_warmup_requests(fake, [
        {"text": "a", "concurrency": 0}, {"text": "b", "concurrency": True},
        {"text": "c", "concurrency": []}, {"text": "d", "concurrency": [2, 0]},
    ])
    assert fake.submitted == []


def test_a_failing_copy_lets_its_siblings_finish(caplog) -> None:
    """One copy failing early is reported after the others end, rather than
    cancelling them mid-request."""
    import asyncio

    class _SlowServer(_FakeServer):
        async def collect_results(self, request_id, raw_request=None):
            if request_id == "warmup-0.0.0":
                raise RuntimeError("compile blew up")
            await asyncio.sleep(0.01)
            self.collected.append(request_id)
            return []

    fake = _SlowServer()
    with caplog.at_level(logging.WARNING):
        entrypoint._run_warmup_requests(fake, [{"text": "a", "concurrency": 3}])
    assert sorted(fake.collected) == ["warmup-0.0.1", "warmup-0.0.2"]
    assert "warmup request 1/1 failed" in caplog.text


def test_api_switch_interval_must_be_a_positive_number(monkeypatch) -> None:
    """``api_switch_interval_s`` (else the model class's default) is refused
    unless a positive number of seconds: yaml's ``true`` is not 1 s."""
    class _Model:
        API_SWITCH_INTERVAL_S = None

    class _Fast:
        API_SWITCH_INTERVAL_S = 0.002

    set_to = []
    monkeypatch.setattr(entrypoint.sys, "setswitchinterval", set_to.append)
    for bad in (True, 0, -0.001, "1ms"):
        with pytest.raises(ValueError, match="api_switch_interval_s"):
            entrypoint._set_switch_interval(_Model, {"api_switch_interval_s": bad})
    entrypoint._set_switch_interval(_Model, {"api_switch_interval_s": 0.001})
    entrypoint._set_switch_interval(_Model, {})
    entrypoint._set_switch_interval(_Fast, {})
    assert set_to == [0.001, 0.002]
