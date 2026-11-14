"""CPU wire checks for the optional Praxis e2e matrix cases."""

from __future__ import annotations

import asyncio
import time
from argparse import Namespace

import aiohttp
import numpy as np
import pytest
from aiohttp import web

from benchmark.eval.datasets import Item
from benchmark.eval.e2e import arm, checks
from benchmark.eval.e2e.manifest import load
from benchmark.eval.suites.e2e import E2ESuite
from benchmark.eval.suites.praxis_wire import (
    ProtocolError,
    _commands,
    _openpi_observation,
    _positronic_observation,
    _ready,
    pack,
    unpack,
)


def _suite(kind: str) -> E2ESuite:
    return E2ESuite(Namespace(kind=kind, media=None, max_tokens=96, audio_out=False,
                              size="1024x1024", extra=[], sample_rate=24000, timeout=5.0))


async def _serve(app: web.Application):
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"


def test_optional_praxis_profiles_and_frontend_args():
    models = load()
    assert not any(m.name.startswith("praxis") for m in models.select("all"))
    rust, python = models.select("tag:praxis")
    assert [c.kind for c in rust.cases] == ["praxis_openpi", "praxis_positronic"]
    assert [c.perf["latency"] for c in rust.cases] == ["lat_p50_s", "warm1_p50_s"]
    assert python.serve_args == ["--python-frontend"]
    original = arm.job
    try:
        arm.job = {"model": python.to_dict()}
        assert arm.serve_args({"rust_server": "unused"}) == ["--python-frontend"]
        arm.job = {"model": rust.to_dict()}
        assert arm.serve_args({"rust_server": "built-server"}) == ["--rust-frontend-bin", "built-server"]
    finally:
        arm.job = original


def test_positronic_probe_refuses_missing_stack_or_wrong_warm_rows():
    with pytest.raises(ProtocolError, match="client stack"):
        _ready({"status": "ready", "protocol_version": 2, "session_id": "a" * 32,
                "meta": {"sample_seed_base": 1234, "chunk_rows": 24, "action_fps": 15.0,
                         "image_codecs": ["h264", "jpeg"], "compress_images": True}}, 0)
    command = {"robot_command": {"type": "cartesian_pos", "pose": np.arange(12, dtype=np.float32)},
               "target_grip": 0.5, "timestamp": 0.0}
    with pytest.raises(ProtocolError, match="not 24"):
        _commands({"result": [command] * 21}, 24)


def test_cold_openpi_payload_passes_the_real_policy_parser():
    from mstar.api_server.openpi import OpenpiPolicy
    from mstar.api_server.openpi import unpack as openpi_unpack

    policy = OpenpiPolicy(
        ("observation/exterior_image_1_left", "observation/wrist_image_left"),
        ("observation/cartesian_position", "observation/gripper_position"),
        "robot_state", 21, 10, 10,
    )
    request = policy.request(openpi_unpack(_openpi_observation(0)))
    assert len(request.images) == 2
    assert [tuple(view.shape) for view in request.images] == [(3, 234, 416)] * 2
    assert request.model_kwargs["robot_state"] == [0, 0, pytest.approx(0.6), 1, 0, 0, 0, 0.5]
    assert request.model_kwargs["seed"] == 1234


def test_warm_positronic_payload_passes_the_real_e5_decoder_and_ready_config():
    """Runs when E5 is in the checkout; main does not have that module or config yet."""
    from pathlib import Path

    import yaml

    decoder = pytest.importorskip("mstar.api_server.positronic_input")
    config_path = Path(__file__).parents[2] / "configs/praxis1_droid.yaml"
    if not config_path.exists():
        pytest.skip("Praxis DROID config is not on this branch")
    meta = yaml.safe_load(config_path.read_text())["positronic"]["meta"]
    _ready({"status": "ready", "protocol_version": 2, "session_id": "a" * 32,
            "meta": {**meta, "sample_seed_base": 1234}}, 0)
    for call, prior, reset in ((0, 0, False), (1, 21, False), (2, 24, False), (3, 0, True)):
        frame = pack({"session_id": "a" * 32, "observation": _positronic_observation(0, call, reset)})
        parsed = decoder.decode_praxis_frame(frame, "single_arm", prior)
        assert not parsed.reset_episode and parsed.executed_rows == prior
        assert [tuple(x.shape) for x in parsed.executed_frames] == [(prior, 3, 234, 416)] * 2
        assert parsed.executed_poses.shape == (prior, 8)


def test_cold_openpi_probe_saves_21x10_actions(tmp_path):
    seen = []

    async def handle(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_bytes(pack({"checkpoint": "runwayml/praxis-1-droid"}))
        async for msg in ws:
            obs = unpack(msg.data)
            seen.append(obs)
            await ws.send_bytes(pack({"actions": np.full((21, 10), 0.25, np.float32)}))
        return ws

    async def run():
        app = web.Application()
        app.router.add_get("/", handle)
        runner, url = await _serve(app)
        try:
            async with aiohttp.ClientSession() as session:
                return await _suite("praxis_openpi").request(session, url, 0, Item(id="000"), time.perf_counter())
        finally:
            await runner.cleanup()

    row = asyncio.run(run())
    assert row["ok"] and row["n_chunks"] == 21 and len(row["_blob"]) == 21 * 10 * 4
    assert seen[0]["observation/exterior_image_1_left"].shape == (234, 416, 3)
    assert seen[0]["seed"] == 1234
    d = tmp_path / "saved"
    _suite("praxis_openpi").score([Item(id="000")], [row], 2.0, d)
    assert (d / "000.f32").exists()


def test_positronic_probe_checks_warm_reset_and_phase_latency(tmp_path):
    seen = []

    async def handle(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        sid = "a" * 32
        await ws.send_bytes(pack({"status": "loading"}))
        await ws.send_bytes(pack({
            "status": "ready", "protocol_version": 2, "session_id": sid,
            "meta": {"sample_seed_base": 1234, "chunk_rows": 24, "action_fps": 15.0,
                     "image_codecs": ["h264", "jpeg"], "compress_images": True,
                     "local_stack": {"seq": [
                         {"name": "stop_on_fault"},
                         {"name": "restrict_image_size", "args": {"width": 416, "height": 234}},
                         {"name": "temporal_stack", "args": {
                             "keys": ["image.wrist", "image.exterior", "robot_state.ee_pose", "grip"],
                             "offsets_sec": [0.0 if frame == 0 else -frame / 15 + 0.5 / 15
                                             for frame in range(24, -1, -1)], "pad_start": False,
                         }},
                         {"name": "chunked_schedule"},
                     ]}}
        }))
        async for msg in ws:
            frame = unpack(msg.data)
            assert frame["session_id"] == sid
            if frame.get("end_session"):
                await ws.send_bytes(pack({"session_id": sid, "end_session": True}))
                break
            obs = frame["observation"]
            seen.append(obs)
            count = 21 if len(seen) in (1, 4) else 24
            command = {"robot_command": {"type": "cartesian_pos", "pose": np.arange(12, dtype=np.float32)},
                       "target_grip": 0.5, "timestamp": 0.0}
            await ws.send_bytes(pack({"result": [command] * count}))
        return ws

    async def run():
        app = web.Application()
        app.router.add_get("/api/v1/session", handle)
        runner, url = await _serve(app)
        try:
            async with aiohttp.ClientSession() as session:
                return await _suite("praxis_positronic").request(session, url, 0, Item(id="000"), time.perf_counter())
        finally:
            await runner.cleanup()

    row = asyncio.run(run())
    assert row["ok"] and row["n_chunks"] == 90 and len(row["_blob"]) == 90 * 14 * 4
    assert [obs.get("reset_context", 0) for obs in seen] == [0, 0, 0, 1]
    assert all(obs["robot_state.ee_pose"].shape == (25, 7) for obs in seen)
    assert all(len(obs["image.exterior"][b"frames"]) == 25 for obs in seen)
    metrics = _suite("praxis_positronic").score([Item(id="000")], [row], 2.0, tmp_path / "saved")
    assert all(metrics[f"{phase}_p50_s"] > 0 for phase in ("cold", "warm1", "warm2", "reset"))
    assert metrics["call_per_s"] == 2.0
    assert checks.saved(tmp_path / "saved", tmp_path / "saved")["action_max_abs_diff"] == 0
