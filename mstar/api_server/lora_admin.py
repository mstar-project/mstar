"""``POST /v1/load_lora_adapter`` and ``POST /v1/unload_lora_adapter``, as vLLM serves them.

A load is checked here first (against the deployment's ``LoraSpec``: LoRA layers present,
rank within ``max_rank``, only declared targets, shapes that fit them) and its weights'
digest taken, then sent with the digest through the conductor to every worker hosting
the LoRA pool. The name becomes selectable, under its digest (``ServedModels``), only
once the conductor reports that every worker loaded it.
An unload takes the name out of selection first, then tells the workers, whose running
requests on it finish on its weights; the name is selectable again if the conductor
refused. This process and ``ServedModels`` are the registry of what is served: the
conductor keeps none.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from fastapi import HTTPException

from mstar.api_server.request_types import LoraAdapterResult
from mstar.api_server.served_models import ServedModels
from mstar.engine.resources.lora.config import (
    LoraSpec,
    describe_load_error,
    resolve_adapter_path,
)
from mstar.engine.resources.lora.store import check_adapter_file
from mstar.utils.ipc_format import (
    ConductorMessage,
    ConductorMessageType,
    LoadLoraAdapter,
    UnloadLoraAdapter,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _Op:
    name: str
    is_load: bool
    # the weights its requests are selected under: a load's once it is served, an
    # unload's restored with the name if the conductor refuses
    digest: bytes
    future: concurrent.futures.Future


class LoraAdmin:
    """``spec`` is the deployment's ``LoraSpec`` (None without LoRA), which a load is
    checked against; ``send`` hands a ``ConductorMessage`` to the thread that sends this
    process's messages to the conductor. ``on_result`` is called from the thread that
    receives the conductor's."""

    def __init__(
        self,
        spec: LoraSpec | None,
        served: ServedModels,
        send: Callable[[ConductorMessage], None],
        timeout_s: float = 300.0,
    ):
        self._spec = spec
        self._served = served
        self._send = send
        self._timeout_s = timeout_s
        # one load or unload at a time, so "already loaded" and "not found" hold until
        # the conductor answers
        self._serial = asyncio.Lock()
        self._ops: dict[str, _Op] = {}
        self._ops_lock = threading.Lock()

    async def load(self, name: str, path: str) -> str:
        self._require_runtime_updates()
        if not name or not path:
            raise ValueError("Both 'lora_name' and 'lora_path' must be non-empty.")
        if name in self._served.base:
            raise ValueError(f"'{name}' is the base model's name")
        async with self._serial:
            self._refuse_unanswered(name)
            if name in self._served.adapters:
                raise ValueError(f"The lora adapter '{name}' has already been loaded.")
            path = resolve_adapter_path(path)
            try:
                digest = await asyncio.to_thread(check_adapter_file, self._spec, name, path)
            except Exception as exc:  # noqa: BLE001 - any unreadable or unfit file is the client's
                # the client sees why the adapter was refused, never what the server's
                # files hold or whether a path exists
                logger.warning("LoRA adapter %s at %s refused: %s", name, path, exc)
                raise ValueError(f"LoRA adapter '{name}': {describe_load_error(exc)}") from exc
            await self._request(
                LoadLoraAdapter(
                    op_id=_op_id(), lora_name=name, lora_path=path, lora_digest=digest.hex(),
                ),
                ConductorMessageType.LOAD_LORA_ADAPTER, digest,
            )
        return f"Success: LoRA adapter '{name}' added successfully."

    async def unload(self, name: str) -> str:
        self._require_runtime_updates()
        if not name:
            raise ValueError("'lora_name' must be non-empty.")
        async with self._serial:
            self._refuse_unanswered(name)
            # Out of selection before any worker hears of it, so no new request names it
            # while they forget it; a request already selected carries the digest it took
            # then. A refusal restores the name as it was (``on_result``).
            digest = self._served.remove(name)
            if digest is None:
                raise HTTPException(
                    status_code=404, detail=f"The lora adapter '{name}' cannot be found.",
                )
            await self._request(
                UnloadLoraAdapter(op_id=_op_id(), lora_name=name),
                ConductorMessageType.UNLOAD_LORA_ADAPTER, digest,
            )
        return f"Success: LoRA adapter '{name}' removed successfully."

    def on_result(self, body: LoraAdapterResult) -> None:
        """The conductor's answer, applied here even when its HTTP request has timed out,
        so this process agrees with the workers: a successful load becomes selectable,
        and a refused unload (one the conductor never sent on) selectable again."""
        with self._ops_lock:
            op = self._ops.pop(body.op_id, None)
        if op is None:
            logger.warning("LoRA: an answer for unknown operation %s", body.op_id)
            return
        if op.is_load and body.error_message is None:
            self._served.add(op.name, op.digest)
            logger.info("LoRA adapter %s is now served", op.name)
        elif not op.is_load and body.error_message is not None:
            self._served.add(op.name, op.digest)
            logger.info("LoRA adapter %s: unload refused (%s); still served", op.name, body.error_message)
        try:
            op.future.set_result(body)
        except concurrent.futures.InvalidStateError:
            pass  # its waiter timed out

    def _require_runtime_updates(self) -> None:
        """Off unless the serving config opts in (``resources: lora: runtime_updates:
        true``): the endpoints make the server read paths a client names. Off, they are
        not there, as vLLM's are not without ``VLLM_ALLOW_RUNTIME_LORA_UPDATING``."""
        if self._spec is None or not self._spec.runtime_updates:
            raise HTTPException(status_code=404, detail="Not Found")

    def _refuse_unanswered(self, name: str) -> None:
        """An earlier operation on ``name`` timed out here and the conductor has not
        answered it yet; going ahead could leave this process and the conductor
        disagreeing about whether the adapter is served."""
        with self._ops_lock:
            if any(op.name == name for op in self._ops.values()):
                raise HTTPException(
                    status_code=409,
                    detail=f"an earlier load or unload of LoRA adapter '{name}' is still in progress",
                )

    async def _request(
        self, body: LoadLoraAdapter | UnloadLoraAdapter, message_type: ConductorMessageType,
        digest: bytes,
    ) -> None:
        future: concurrent.futures.Future = concurrent.futures.Future()
        with self._ops_lock:
            self._ops[body.op_id] = _Op(
                name=body.lora_name, is_load=isinstance(body, LoadLoraAdapter),
                digest=digest, future=future,
            )
        self._send(ConductorMessage(message_type=message_type, body=body))
        try:
            result: LoraAdapterResult = await asyncio.wait_for(
                asyncio.wrap_future(future), self._timeout_s,
            )
        except asyncio.TimeoutError as exc:
            raise HTTPException(
                status_code=504,
                detail=f"no answer from the workers within {self._timeout_s:.0f} s; "
                f"LoRA adapter '{body.lora_name}' may still be "
                f"{'loaded' if isinstance(body, LoadLoraAdapter) else 'unloaded'} later",
            ) from exc
        if result.error_message is not None:
            raise HTTPException(status_code=result.status, detail=result.error_message)


def _op_id() -> str:
    return f"lora-{uuid.uuid4().hex}"
