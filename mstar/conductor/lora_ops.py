"""Loading and unloading LoRA adapters at runtime, as the conductor sees it.

The API server asks; the conductor forwards the request to every worker hosting a node
with the LoRA pool and answers once all of them have acknowledged. A load becomes
servable (the conductor's ``LoraSpec`` names it, so ``LoraReqConfig.validate`` accepts it)
only when every worker loaded it; if any failed, the others are told to unload it again.
An unload stops the conductor validating new requests on the name at once, before any
worker hears of it.

A worker that never answers (hung, dead) would leave the name refused with a 409 for good,
on this side and the API server's. Past ``runtime_update_timeout_s`` an operation is given
up: a load is undone on every worker and answered with a 504, an unload (the name already
out of the conductor's spec) answered as done.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from mstar.api_server.request_types import APIServerMessage, LoraAdapterResult
from mstar.engine.resources.lora.config import LoraSpec
from mstar.utils.ipc_format import (
    LoadLoraAdapter,
    LoraAdapterAck,
    UnloadLoraAdapter,
    WorkerMessage,
    WorkerMessageType,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _PendingOp:
    """A load or unload sent to the workers and not yet acknowledged by all of them."""

    body: LoadLoraAdapter | UnloadLoraAdapter
    waiting: set[str]
    # the conductor's clock past which the workers still waited on are given up
    deadline: float
    # worker id -> its error
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def is_load(self) -> bool:
        return isinstance(self.body, LoadLoraAdapter)


class LoraAdapterOps:
    """The conductor's half of ``/v1/load_lora_adapter`` and ``/v1/unload_lora_adapter``.

    ``spec`` is the conductor's own ``LoraSpec`` (None for a deployment without LoRA), the
    one requests are validated against, and ``others`` the model's LoRA specs for other
    nodes, which serve the same adapters; ``workers`` are the workers hosting any of their
    nodes; ``send(peer, message)`` is the conductor's communicator; ``clock`` its
    monotonic time.
    """

    def __init__(
        self, spec: LoraSpec | None, workers: list[str], send: Callable[[str, object], None],
        clock: Callable[[], float] = time.monotonic,
        others: Sequence[LoraSpec] = (),
    ):
        self._spec = spec
        self._others = tuple(others)
        self._workers = list(workers)
        self._send = send
        self._clock = clock
        self._pending: dict[str, _PendingOp] = {}

    def load(self, body: LoadLoraAdapter) -> None:
        refusal = self._refusal(body.lora_name)
        if refusal is None and body.lora_name in self._spec.adapters:
            refusal = 400, f"The lora adapter '{body.lora_name}' has already been loaded."
        if refusal is not None:
            self._answer(body.op_id, *refusal, name=body.lora_name)
            return
        self._start(body, WorkerMessageType.LOAD_LORA_ADAPTER)

    def unload(self, body: UnloadLoraAdapter) -> None:
        refusal = self._refusal(body.lora_name)
        if refusal is None and body.lora_name not in self._spec.adapters:
            refusal = 404, f"The lora adapter '{body.lora_name}' cannot be found."
        if refusal is not None:
            self._answer(body.op_id, *refusal, name=body.lora_name)
            return
        # refuse new requests on it from here on; running ones finish on its weights
        for spec in (self._spec, *self._others):
            spec.remove_adapter(body.lora_name)
        self._start(body, WorkerMessageType.UNLOAD_LORA_ADAPTER)

    def ack(self, body: LoraAdapterAck) -> None:
        op = self._pending.get(body.op_id)
        if op is None:
            if body.error is not None:
                logger.warning("LoRA: %s on %s failed: %s", body.op_id, body.worker_id, body.error)
            return  # a rollback's ack: nothing waits on it
        op.waiting.discard(body.worker_id)
        if body.error is not None:
            op.errors[body.worker_id] = body.error
        if not op.waiting:
            del self._pending[body.op_id]
            self._finish(op)

    def _refusal(self, name: str) -> tuple[int, str] | None:
        if self._spec is None or not self._spec.runtime_updates:
            return 404, "Not Found"  # as the API server answers
        if any(op.body.lora_name == name for op in self._pending.values()):
            return 409, f"a load or unload of LoRA adapter '{name}' is already in progress"
        return None

    def expire(self) -> None:
        """Give up on the operations some worker has not answered within
        ``runtime_update_timeout_s``. A late ack for one is ignored."""
        if not self._pending:
            return
        now = self._clock()
        for op_id, op in list(self._pending.items()):
            if op.deadline > now:
                continue
            del self._pending[op_id]
            name = op.body.lora_name
            logger.error(
                "LoRA %s of %s: no answer from %s within %.0f s; giving up on it",
                "load" if op.is_load else "unload", name, sorted(op.waiting),
                self._spec.runtime_update_timeout_s,
            )
            if not op.is_load:
                # out of the spec since it started: new requests are refused, which is
                # what the client asked for; a silent worker forgets it, or dies
                self._answer(op_id, 200, name=name)
                continue
            # every worker, the silent ones included: a load still being read there is
            # undone once it lands (the worker holds the unload behind it)
            self._roll_back(op, self._workers)
            self._answer(
                op_id, 504,
                f"LoRA adapter '{name}' was not loaded: {len(op.waiting)} worker(s) did not "
                f"answer within {self._spec.runtime_update_timeout_s:.0f} s",
                name=name,
            )

    def _start(self, body: LoadLoraAdapter | UnloadLoraAdapter, message_type: WorkerMessageType) -> None:
        op = _PendingOp(
            body=body, waiting=set(self._workers),
            deadline=self._clock() + self._spec.runtime_update_timeout_s,
        )
        if not op.waiting:
            self._finish(op)
            return
        self._pending[body.op_id] = op
        for worker_id in self._workers:
            self._send(worker_id, WorkerMessage(message_type=message_type, body=body))

    def _finish(self, op: _PendingOp) -> None:
        name = op.body.lora_name
        if not op.is_load:
            if op.errors:
                logger.warning("LoRA adapter %s: unload failed on %s", name, op.errors)
            logger.info("LoRA adapter %s unloaded", name)
            self._answer(op.body.op_id, 200, name=name)
            return
        if not op.errors:
            digest = op.body.lora_digest
            for spec in (self._spec, *self._others):
                spec.add_adapter(
                    name, op.body.lora_path, None if digest is None else bytes.fromhex(digest),
                )
            logger.info("LoRA adapter %s loaded from %s", name, op.body.lora_path)
            self._answer(op.body.op_id, 200, name=name)
            return
        self._roll_back(op, sorted(set(self._workers) - set(op.errors)))
        # each worker logged its own failure in full; what they ack is fit for a client
        logger.warning("LoRA adapter %s failed to load on %s", name, op.errors)
        detail = "; ".join(sorted(set(op.errors.values())))
        self._answer(
            op.body.op_id, 400, f"LoRA adapter '{name}' failed to load: {detail}", name=name,
        )

    def _roll_back(self, op: _PendingOp, workers) -> None:
        """Have ``workers`` forget a load. Nothing can be running on it, so each forgets
        at once, and a later load of the name reaches each after this."""
        rollback = UnloadLoraAdapter(op_id=f"{op.body.op_id}:rollback", lora_name=op.body.lora_name)
        for worker_id in workers:
            self._send(worker_id, WorkerMessage(
                message_type=WorkerMessageType.UNLOAD_LORA_ADAPTER, body=rollback,
            ))

    def _answer(self, op_id: str, status: int, error: str | None = None, name: str | None = None) -> None:
        served = None if self._spec is None or name is None else name in self._spec.adapters
        self._send("api_server", APIServerMessage(
            message_type="lora_adapter_result",
            body=LoraAdapterResult(op_id=op_id, error_message=error, status=status, served=served),
        ))
