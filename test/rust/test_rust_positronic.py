"""Positronic v2 stub sessions send the same requests through both frontends."""

import asyncio
import contextlib
import json
import os
import re
import subprocess
import tempfile
import threading
import time
from types import SimpleNamespace

import msgpack
import numpy as np
import pytest
import websockets.exceptions
import websockets.sync.client
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from test_rust_frontend import BINARY, _free_port, _wait_healthy

from mstar.api_server import entrypoint, openpi
from mstar.api_server.positronic import PositronicPolicy
from mstar.api_server.positronic_input import decode_praxis_frame
from mstar.api_server.request_types import ResultChunk
from mstar.api_server.rust_frontend import RustFrontendBridge, positronic_facts

pytestmark = pytest.mark.skipif(BINARY is None, reason="mstar-server binary not built")


@pytest.mark.parametrize("meta", [{"bytes": b"x"}, {"nan": float("nan")}])
def test_policy_metadata_must_pass_the_rust_launcher_json(meta):
    with pytest.raises(ValueError, match="meta must be JSON serializable"):
        PositronicPolicy.from_config({"meta": meta}, object())


def test_config_cannot_advertise_ready_without_a_model_session_adapter():
    with pytest.raises(ValueError, match="no ready Positronic session adapter"):
        PositronicPolicy.from_config({"meta": {"robot": "praxis"}}, object())


def test_normal_rust_launcher_drops_an_inherited_session_policy(monkeypatch):
    from mstar.api_server import rust_frontend

    seen = {}
    monkeypatch.setenv("MSTAR_POSITRONIC", '{"meta": {"robot": "unapproved"}}')

    def fake_popen(args, env):
        seen["env"] = env
        return SimpleNamespace(pid=1)

    monkeypatch.setattr(rust_frontend.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(rust_frontend, "soundfile_library", lambda: None)
    monkeypatch.setattr(rust_frontend, "openai_adapter_class", lambda name: "")
    rust_frontend.launch_rust_server("bin", "stub", 12345, "bridge", "uploads", positronic=None)
    assert "MSTAR_POSITRONIC" not in seen["env"]


class Stub:
    positronic_policy = PositronicPolicy({"robot": "stub"})
    metrics = None
    fatal_error = None

    def __init__(self):
        self.upload_dir = tempfile.mkdtemp(prefix="positronic_stub_")
        self.calls = []
        self.submissions = []
        self.ended = []
        self.depth = {}

    def submit_request(self, **kw):
        self.submissions.append(kw)
        episode = kw["episode"]
        reset = kw["reset_episode"]
        if reset:
            self.depth[episode] = 0
        depth = self.depth.get(episode, 0)
        self.calls.append((kw["text"], episode, reset, depth))
        self.depth[episode] = depth + 1
        return f"stub-{len(self.calls)}"

    async def iter_result_chunks(self, request_id):
        await asyncio.sleep(0)
        depth = self.calls[-1][3]
        rows = [
            {"robot_command": {"type": "cartesian_pos", "pose": np.full(12, i, dtype=np.float32)},
             "target_grip": 0.0, "timestamp": i / 15.0}
            for i in range(21 if depth == 0 else 24)
        ]
        yield ResultChunk(request_id, "action", openpi.pack(rows))

    def end_episode(self, episode):
        self.ended.append(episode)
        self.depth.pop(episode, None)

    def submission_key(self, rid):
        return rid

    def relay_chunks_to(self, key, sink, tag):
        pass

    def abort_submission(self, key):
        pass


@contextlib.contextmanager
def rust(stub):
    port = _free_port()
    bridge_dir = tempfile.mkdtemp(prefix="positronic_bridge_")
    env = dict(os.environ, MSTAR_POSITRONIC=json.dumps(positronic_facts(stub.positronic_policy)))
    proc = subprocess.Popen([BINARY, "stub", str(port), bridge_dir, stub.upload_dir], env=env)
    bridge = RustFrontendBridge(stub, bridge_dir)
    threading.Thread(target=bridge.run, daemon=True).start()
    try:
        assert _wait_healthy(port)
        yield port
    finally:
        bridge.stop()
        proc.terminate()
        proc.wait(timeout=10)


def frame(id, **obs):
    return openpi.pack({
        "session_id": id,
        "observation": {"prompt": "go", "state": np.arange(8, dtype=np.float32), **obs},
    })


def _normalise(messages):
    ready = messages[1]
    assert ready["status"] == "ready" and ready["protocol_version"] == 2
    assert re.fullmatch(r"[0-9a-f]{32}", ready["session_id"])
    ready["session_id"] = "ID"
    for m in messages[2:]:
        if "session_id" in m:
            m["session_id"] = "ID"
        if "timing" in m:
            assert set(m["timing"]) == {"served_ms", "decode_ms", "infer_ms", "queued_ms"}
            m["timing"] = "TIMING"
    return messages


def python_session(stub, make_frame=frame, requests=3, reset_last=True):
    entrypoint.api_server = stub
    with TestClient(entrypoint.app).websocket_connect("/api/v1/session") as ws:
        messages = [msgpack.unpackb(ws.receive_bytes()) for _ in range(2)]
        id = messages[1]["session_id"]
        for call_index in range(requests):
            data = make_frame(id, call_index=call_index, **({"reset_context": True}
                                                            if reset_last and call_index == requests - 1 else {}))
            ws.send_bytes(data)
            messages.append(msgpack.unpackb(ws.receive_bytes()))
        ws.send_bytes(msgpack.packb({"session_id": id, "end_session": True}))
        messages.append(msgpack.unpackb(ws.receive_bytes()))
    return messages


def rust_session(port, make_frame=frame, requests=3, reset_last=True):
    with websockets.sync.client.connect(f"ws://127.0.0.1:{port}/api/v1/session", compression=None) as ws:
        messages = [msgpack.unpackb(ws.recv()) for _ in range(2)]
        id = messages[1]["session_id"]
        for call_index in range(requests):
            data = make_frame(id, call_index=call_index, **({"reset_context": True}
                                                            if reset_last and call_index == requests - 1 else {}))
            ws.send(data)
            messages.append(msgpack.unpackb(ws.recv()))
        ws.send(msgpack.packb({"session_id": id, "end_session": True}))
        messages.append(msgpack.unpackb(ws.recv()))
    return messages


def test_cold_warm_reset_and_end_match_in_both_frontends():
    stub = Stub()
    with rust(stub) as port:
        py = python_session(stub)
        py_calls, py_ends = list(stub.calls), list(stub.ended)
        stub.calls.clear()
        rust_reply = rust_session(port)
        rust_calls, rust_ends = list(stub.calls), list(stub.ended[1:])
    assert _normalise(py) == _normalise(rust_reply)
    assert [len(m["result"]) for m in py[2:5]] == [21, 24, 21]
    assert py[5] == {"session_id": "ID", "end_session": True}
    assert len(py_ends) == len(rust_ends) == 1
    assert [c[:1] + c[2:] for c in py_calls] == [c[:1] + c[2:] for c in rust_calls]
    assert all(c[1] == py_ends[0] for c in py_calls)
    assert all(c[1] == rust_ends[0] for c in rust_calls)
    assert set(py[2]) == {"result", "effective_prompt", "prompt_original", "timing"}


def test_praxis_mode_sends_same_original_observation_to_shared_data_worker():
    stub = Stub()
    stub.positronic_policy = PositronicPolicy(
        {"robot": "stub"}, observation_format="praxis1", control_mode="droid_impedance",
    )
    def stacked_frame(session_id, **extra):
        images = np.stack([np.full((4, 6, 3), n, np.uint8) for n in (10, 20, 30, 40)])
        return frame(
            session_id, **{
                "image.exterior": images,
                "image.wrist": images + 1,
                "robot_state.ee_pose": np.array([[0, 0, n, 1, 0, 0, 0] for n in (0.1, 0.2, 0.3, 0.4)], np.float32),
                "grip": np.array([0.1, 0.2, 0.3, 0.4], np.float32),
                "descriptor": "remote.robolab.droid", "executed_rows": 999, **extra,
            },
        )

    with rust(stub) as port:
        python_session(stub, stacked_frame, requests=4)
        py_submissions = list(stub.submissions)
        stub.submissions.clear()
        rust_session(port, stacked_frame, requests=4)
        rs_submissions = list(stub.submissions)
    assert len(py_submissions) == len(rs_submissions) == 4
    for py, rs in zip(py_submissions, rs_submissions, strict=True):
        for submission in (py, rs):
            assert submission["input_modalities"] == ["image", "text"]
            assert submission["model_kwargs"]["control_mode"] == "droid_impedance"
            assert submission["model_kwargs"]["executed_rows"] in (0, 21, 24)
            parsed = msgpack.unpackb(submission["positronic_frame"], raw=False)
            assert parsed["session_id"] == submission["episode"]
            assert parsed["observation"]["prompt"] == "go"
            assert b"__ndarray__" in parsed["observation"]["image.exterior"]
            decoded = decode_praxis_frame(submission["positronic_frame"], "single_arm")
            assert [int(image[0, 0, 0]) for image in decoded.images] == [40, 41]
            assert decoded.robot_state[2] == pytest.approx(0.4)
        assert py["positronic_frame"].replace(py["episode"].encode(), b"ID") == rs["positronic_frame"].replace(
            rs["episode"].encode(), b"ID"
        )
    assert [s["reset_episode"] for s in py_submissions] == [False, False, False, True]
    assert [s["reset_episode"] for s in rs_submissions] == [False, False, False, True]
    assert [s["model_kwargs"]["executed_rows"] for s in py_submissions] == [0, 21, 24, 0]
    assert [s["model_kwargs"]["executed_rows"] for s in rs_submissions] == [0, 21, 24, 0]


def test_short_stack_cold_reply_resynchronizes_both_frontends():
    class WindowStub(Stub):
        def submit_request(self, **kw):
            decoded = decode_praxis_frame(
                kw["positronic_frame"], "single_arm", kw["model_kwargs"]["executed_rows"],
            )
            if decoded.reset_episode:
                self.depth[kw["episode"]] = 0
            return super().submit_request(**kw)

    def window_frame(session_id, call_index, **extra):
        count = (1, 22, 21, 22)[call_index]
        frames = np.stack([np.full((3, 4, 3), i, np.uint8) for i in range(count)])
        return frame(session_id, **{
            "image.exterior": frames, "image.wrist": frames,
            "robot_state.ee_pose": np.array([[i, 0, 0.6, 1, 0, 0, 0] for i in range(count)], np.float32),
            "grip": np.zeros(count, np.float32), **extra,
        })

    stub = WindowStub()
    stub.positronic_policy = PositronicPolicy({"robot": "stub"}, observation_format="praxis1")
    with rust(stub) as port:
        py = python_session(stub, window_frame, requests=4, reset_last=False)
        py_submissions = list(stub.submissions)
        stub.submissions.clear()
        rs = rust_session(port, window_frame, requests=4, reset_last=False)
        rs_submissions = list(stub.submissions)
    assert _normalise(py) == _normalise(rs)
    assert [len(m["result"]) for m in py[2:6]] == [21, 24, 21, 24]
    for submissions in (py_submissions, rs_submissions):
        assert [s["model_kwargs"]["executed_rows"] for s in submissions] == [0, 21, 24, 21]
        resets = [
            decode_praxis_frame(s["positronic_frame"], "single_arm", s["model_kwargs"]["executed_rows"]).reset_episode
            for s in submissions
        ]
        assert resets == [False, False, True, False]


def test_wrong_session_id_fails_and_ends_once():
    stub = Stub()
    with rust(stub) as port:
        entrypoint.api_server = stub
        with TestClient(entrypoint.app).websocket_connect("/api/v1/session") as ws:
            ws.receive_bytes()
            ws.receive_bytes()
            ws.send_bytes(frame("wrong"))
            py = msgpack.unpackb(ws.receive_bytes())
            with pytest.raises(WebSocketDisconnect) as close:
                ws.receive_bytes()
            assert close.value.code == 1011
        with websockets.sync.client.connect(f"ws://127.0.0.1:{port}/api/v1/session", compression=None) as ws:
            ws.recv()
            ws.recv()
            ws.send(frame("wrong"))
            rs = msgpack.unpackb(ws.recv())
            with pytest.raises(websockets.exceptions.ConnectionClosed) as close:
                ws.recv()
            assert close.value.rcvd.code == 1011
    assert py == rs == {"status": "error", "error": "The session ID does not belong to this connection"}
    assert len(stub.ended) == 2


def test_text_frame_is_refused_and_disconnect_releases_the_episode():
    stub = Stub()
    with rust(stub) as port:
        entrypoint.api_server = stub
        with TestClient(entrypoint.app).websocket_connect("/api/v1/session") as ws:
            ws.receive_bytes()
            py_id = msgpack.unpackb(ws.receive_bytes())["session_id"]
            ws.send_text("bad")
            py = msgpack.unpackb(ws.receive_bytes())
            with pytest.raises(WebSocketDisconnect) as close:
                ws.receive_bytes()
            assert close.value.code == 1008
        with websockets.sync.client.connect(f"ws://127.0.0.1:{port}/api/v1/session", compression=None) as ws:
            ws.recv()
            rust_id = msgpack.unpackb(ws.recv())["session_id"]
            ws.send("bad")
            rs = msgpack.unpackb(ws.recv())
            with pytest.raises(websockets.exceptions.ConnectionClosed) as close:
                ws.recv()
            assert close.value.rcvd.code == 1008
    assert py == rs == {
        "status": "error", "error": "Text frames are not served. Send each observation as one binary msgpack frame.",
    }
    assert stub.ended == [py_id, rust_id]


def test_malformed_binary_frame_fails_in_both_frontends():
    stub = Stub()
    with rust(stub) as port:
        entrypoint.api_server = stub
        with TestClient(entrypoint.app).websocket_connect("/api/v1/session") as ws:
            ws.receive_bytes()
            py_id = msgpack.unpackb(ws.receive_bytes())["session_id"]
            ws.send_bytes(b"\xc1")
            py = msgpack.unpackb(ws.receive_bytes())
            with pytest.raises(WebSocketDisconnect) as close:
                ws.receive_bytes()
            assert close.value.code == 1011
        with websockets.sync.client.connect(f"ws://127.0.0.1:{port}/api/v1/session", compression=None) as ws:
            ws.recv()
            rust_id = msgpack.unpackb(ws.recv())["session_id"]
            ws.send(b"\xc1")
            rs = msgpack.unpackb(ws.recv())
            with pytest.raises(websockets.exceptions.ConnectionClosed) as close:
                ws.recv()
            assert close.value.rcvd.code == 1011
    assert py == rs == {"status": "error", "error": "invalid msgpack request"}
    assert stub.ended == [py_id, rust_id]


def test_closing_without_end_session_releases_each_connection_once():
    stub = Stub()
    with rust(stub) as port:
        entrypoint.api_server = stub
        with TestClient(entrypoint.app).websocket_connect("/api/v1/session") as ws:
            ws.receive_bytes()
            py_id = msgpack.unpackb(ws.receive_bytes())["session_id"]
        with websockets.sync.client.connect(f"ws://127.0.0.1:{port}/api/v1/session", compression=None) as ws:
            ws.recv()
            rust_id = msgpack.unpackb(ws.recv())["session_id"]
        deadline = time.monotonic() + 2
        while len(stub.ended) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
    assert stub.ended == [py_id, rust_id]
