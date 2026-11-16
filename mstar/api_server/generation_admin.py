"""Authenticated HTTP control of the conductor's generation state.

The HTTP frontends share this path so a checkpoint named by a client is
checked against the deployment's trusted roots before it reaches a worker.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import threading
import uuid
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from fastapi import HTTPException

from mstar.api_server.request_types import GenerationControlResult
from mstar.utils.ipc_format import (
    CheckWeights,
    ConductorMessage,
    ConductorMessageType,
    ContinueGeneration,
    GetGenerationState,
    PauseGeneration,
    UpdateWeightsFromDisk,
)

RELOAD_ROOTS_ENV = "MSTAR_WEIGHT_RELOAD_ROOTS"


def reload_roots(value: str | None = None) -> tuple[Path, ...]:
    """Existing, absolute, non-root directories trusted for checkpoint reloads."""
    text = os.environ.get(RELOAD_ROOTS_ENV, "") if value is None else value
    roots = []
    for part in filter(None, text.split(os.pathsep)):
        path = Path(part)
        if not path.is_absolute():
            raise ValueError(f"{RELOAD_ROOTS_ENV} entries must be absolute directories")
        root = path.resolve(strict=True)
        if not root.is_dir() or root == Path(root.anchor):
            raise ValueError(f"{RELOAD_ROOTS_ENV} must contain non-root directories")
        roots.append(root)
    return tuple(roots)


class GenerationAdmin:
    """Request/answer bridge for #845's existing conductor control messages."""

    def __init__(
        self, send: Callable[[ConductorMessage], None], *, key_enabled: bool,
        roots: tuple[Path, ...], timeout_s: float = 600.0,
    ):
        self._send = send
        self._enabled = key_enabled and bool(roots)
        self._roots = roots
        self._timeout_s = timeout_s
        self._pending: dict[str, concurrent.futures.Future[GenerationControlResult]] = {}
        self._lock = threading.Lock()

    def _require_enabled(self) -> None:
        if not self._enabled:
            raise HTTPException(status_code=404, detail="Not Found")

    def _checkpoint(self, model_path: str) -> str:
        if not isinstance(model_path, str) or not model_path or not Path(model_path).is_absolute():
            raise ValueError("model_path must name an absolute checkpoint directory")
        try:
            path = Path(model_path).resolve(strict=True)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError("model_path is not an allowed checkpoint directory") from exc
        if not path.is_dir() or not any(path == root or path.is_relative_to(root) for root in self._roots):
            raise ValueError("model_path is not an allowed checkpoint directory")
        return str(path)

    async def call(self, op: str, body: dict | None = None) -> dict:
        self._require_enabled()
        body = body or {}
        op_id = f"generation-{uuid.uuid4().hex}"
        if op == "pause_generation":
            mode = body.get("mode", "abort")
            if mode not in ("abort", "wait"):
                raise ValueError("mode must be 'abort' or 'wait'")
            message_type, payload = ConductorMessageType.PAUSE_GENERATION, PauseGeneration(op_id, mode)
        elif op == "continue_generation":
            message_type, payload = ConductorMessageType.CONTINUE_GENERATION, ContinueGeneration(op_id)
        elif op == "update_weights_from_disk":
            version = body.get("weight_version")
            if not isinstance(version, str) or not version:
                raise ValueError("weight_version must be a non-empty string")
            path = self._checkpoint(body.get("model_path"))
            message_type = ConductorMessageType.UPDATE_WEIGHTS_FROM_DISK
            payload = UpdateWeightsFromDisk(op_id, path, version)
        elif op == "is_paused":
            message_type, payload = ConductorMessageType.GET_GENERATION_STATE, GetGenerationState(op_id)
        elif op == "weight_checker":
            mode = body.get("mode", "checksum")
            if mode not in ("checksum", "compare"):
                raise ValueError("mode must be 'checksum' or 'compare'")
            checksums = body.get("checksums", {})
            if not isinstance(checksums, dict) or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in checksums.items()
            ):
                raise ValueError("checksums must map rank names to SHA-256 hex digests")
            message_type = ConductorMessageType.CHECK_WEIGHTS
            payload = CheckWeights(op_id=op_id, mode=mode, checksums=checksums)
        else:
            raise ValueError("unknown generation operation")

        future: concurrent.futures.Future[GenerationControlResult] = concurrent.futures.Future()
        with self._lock:
            self._pending[op_id] = future
        try:
            self._send(ConductorMessage(message_type=message_type, body=payload))
            result = await asyncio.wait_for(asyncio.wrap_future(future), self._timeout_s)
        except asyncio.TimeoutError as exc:
            raise HTTPException(status_code=504, detail="no conductor answer; generation state is unknown") from exc
        finally:
            with self._lock:
                self._pending.pop(op_id, None)
        return asdict(result)

    def on_result(self, result: GenerationControlResult) -> None:
        with self._lock:
            future = self._pending.get(result.op_id)
        if future is not None:
            try:
                future.set_result(result)
            except concurrent.futures.InvalidStateError:
                pass  # the HTTP waiter timed out as this answer arrived
