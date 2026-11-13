"""Positronic protocol v2 session wire.

The model adapter supplies command dictionaries as a msgpack ``action`` chunk.
Praxis observation decoding and pose composition belong to its model adapter.
"""

from __future__ import annotations

import json
import logging
import secrets
import time
from dataclasses import dataclass

import msgpack
from fastapi import WebSocket, WebSocketDisconnect

from mstar.api_server.openpi import pack as _pack_numpy
from mstar.api_server.openpi import unpack as _unpack_numpy
from mstar.utils.client_errors import client_error_text

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PositronicPolicy:
    meta: dict
    prompt_key: str = "prompt"

    @classmethod
    def from_config(cls, block: dict, model: object) -> PositronicPolicy:
        if not isinstance(block, dict) or set(block) - {"meta", "prompt_key"}:
            raise ValueError("positronic: takes only meta and prompt_key")
        meta = block.get("meta", {})
        prompt_key = block.get("prompt_key", "prompt")
        if not isinstance(meta, dict) or not isinstance(prompt_key, str) or not prompt_key:
            raise ValueError("positronic: meta must be a map and prompt_key a nonempty string")
        try:
            json.dumps(meta, allow_nan=False)
        except (TypeError, ValueError):
            raise ValueError("positronic: meta must be JSON serializable") from None
        if not getattr(model, "POSITRONIC_SESSION_READY", False):
            raise ValueError("positronic: this model has no ready Positronic session adapter")
        return cls(meta, prompt_key)


def _pack(value: dict) -> bytes:
    return _pack_numpy(value)


def _request(data: bytes, session_id: str) -> tuple[dict | None, bool]:
    try:
        value = msgpack.unpackb(data, raw=False, strict_map_key=False)
    except (ValueError, TypeError, msgpack.UnpackException):
        raise ValueError("invalid msgpack request") from None
    if not isinstance(value, dict) or value.get("session_id") != session_id:
        raise ValueError("The session ID does not belong to this connection")
    if value.get("end_session") is True:
        return None, True
    obs = value.get("observation")
    if not isinstance(obs, dict):
        raise ValueError("a v2 request carries an observation dict")
    return obs, False


def _prompt(obs: dict, policy: PositronicPolicy) -> str:
    prompt = obs.get(policy.prompt_key)
    if not isinstance(prompt, str):
        raise ValueError(f"the observation needs a {policy.prompt_key} string")
    return prompt


async def _infer(api, obs: dict, policy: PositronicPolicy, session_id: str, reset: bool) -> list[dict]:
    prompt = _prompt(obs, policy)
    request_id = api.submit_request(
        text=prompt, input_modalities=["text"], output_modalities=["action"],
        streaming=True, episode=session_id, reset_episode=reset,
    )
    data = None
    async for chunk in api.iter_result_chunks(request_id):
        if chunk.modality == "error":
            raise ValueError(bytes(chunk.data).decode("utf-8", "replace"))
        if chunk.modality == "action":
            data = bytes(chunk.data)
    if data is None:
        raise ValueError("the request finished without actions")
    try:
        rows = _unpack_numpy(data)
    except (ValueError, TypeError, msgpack.UnpackException):
        raise ValueError("invalid command result") from None
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("the model must return a list of command dictionaries")
    return rows


async def serve_connection(websocket: WebSocket, api) -> None:
    """One connection owns one episode, ended once on every exit path."""
    policy = getattr(api, "positronic_policy", None)
    if policy is None:
        await websocket.close(code=1008, reason="this server has no positronic policy")
        return
    await websocket.accept()
    session_id = secrets.token_hex(16)
    ended = False
    try:
        await websocket.send_bytes(_pack({"status": "loading", "message": "Loading policy..."}))
        await websocket.send_bytes(_pack({
            "status": "ready", "meta": policy.meta, "protocol_version": 2,
            "session_id": session_id, "max_in_flight": 1,
        }))
        while True:
            frame = await websocket.receive()
            if frame["type"] == "websocket.disconnect":
                return
            data = frame.get("bytes")
            if data is None:
                reason = "Text frames are not served. Send each observation as one binary msgpack frame."
                await websocket.send_bytes(_pack({
                    "status": "error",
                    "error": reason,
                }))
                await websocket.close(code=1008, reason=reason[:100])
                return
            started = time.monotonic()
            try:
                obs, end = _request(data, session_id)
                if end:
                    ended = True
                    api.end_episode(session_id)
                    await websocket.send_bytes(_pack({"session_id": session_id, "end_session": True}))
                    return
                assert obs is not None
                reset = obs.get("reset_context") is True or obs.get("__reset_context__") is True
                infer_started = time.monotonic()
                rows = await _infer(api, obs, policy, session_id, reset)
                elapsed = (time.monotonic() - started) * 1000
                infer_ms = (time.monotonic() - infer_started) * 1000
                await websocket.send_bytes(_pack({
                    "result": rows, "effective_prompt": _prompt(obs, policy), "prompt_original": None,
                    "timing": {"served_ms": round(elapsed, 1), "decode_ms": round((infer_started - started) * 1000, 1),
                               "infer_ms": round(infer_ms, 1), "queued_ms": 0.0},
                }))
            except Exception as exc:  # a call may have changed retained state
                text, _ = client_error_text(exc, "positronic request failed", log=logger, type_name=False)
                await websocket.send_bytes(_pack({"status": "error", "error": text}))
                await websocket.close(code=1011)
                return
    except WebSocketDisconnect:
        pass
    finally:
        if not ended:
            api.end_episode(session_id)
