"""Mooncake Store as a KV store backend.

One client per KV cache with a store. The host pool's tensor is registered once, and each page is one store
object made of its per-layer blocks: a page is one block per layer, a layer's stride apart, which
Mooncake's multi-buffer calls read and write without a staging copy.
"""

from __future__ import annotations

import logging
import os
import re
import socket
import threading
from contextlib import contextmanager

import torch

from mstar.engine.resources.kv.store.base import GetResult, KVStoreBackend
from mstar.engine.resources.kv.store.config import MooncakeStoreConfig

logger = logging.getLogger(__name__)

# Mooncake's staging buffer for copies from memory that isn't registered; every page here is registered
_LOCAL_BUFFER_BYTES = 64 << 20
# Get codes Mooncake returns before any transfer starts: INVALID_PARAMS (a buffer of the wrong size, say),
# REPLICA_IS_NOT_READY, OBJECT_NOT_FOUND, REPLICA_NOT_FOUND. Every other error may follow a transfer.
_NOT_THERE = frozenset({-600, -703, -704, -710})
# Mooncake's TCP transport queues at most this many transfers per peer (a page is one transfer per layer). A
# call past it fails after a 60 s stall and leaves the client failing later transfers, so TCP calls are split
# to a quarter of it: room for a put beside a get.
_TCP_QUEUE_DEFAULT = 2048  # 0.3.13's: a call of 2,072 transfers failed there, one of 2,044 didn't


def _tcp_queue() -> int:
    """``MC_TCP_MAX_QUEUED_TRANSFERS_PER_PEER`` as a positive number, else Mooncake's default."""
    match = re.match(r"\s*(\d+)", os.environ.get("MC_TCP_MAX_QUEUED_TRANSFERS_PER_PEER", ""))
    return int(match.group(1)) if match and int(match.group(1)) > 0 else _TCP_QUEUE_DEFAULT


class MooncakeStore(KVStoreBackend):
    def __init__(self, config: MooncakeStoreConfig):
        try:
            from mooncake.store import MooncakeDistributedStore
        except ImportError as e:
            from mstar.communication.tensors import _mooncake_install_hint

            raise RuntimeError(
                f"The KV store's mooncake backend can't load Mooncake: {e}. "
                + _mooncake_install_hint(e, torch.version.cuda, alternative=None)
            ) from e
        self._client_type = MooncakeDistributedStore
        self._config = config
        self._lock = _SharedLock()
        self._store = self._connect()
        self._tcp = config.protocol == "tcp"
        self._base = 0
        self._registered = 0

    def _connect(self):
        config = self._config
        client = self._client_type()
        hostname = config.local_hostname or _default_hostname(config.master)
        rc = client.setup(
            hostname, config.metadata, config.segment_bytes, _LOCAL_BUFFER_BYTES,
            config.protocol, config.rdma_devices, config.master,
        )
        if rc != 0:
            raise RuntimeError(
                f"Mooncake Store setup failed with {rc}: master {config.master!r}, metadata "
                f"{config.metadata!r}, protocol {config.protocol!r}, this host {hostname!r}"
            )
        return client

    def bind(self, host_pages: torch.Tensor) -> None:
        assert host_pages.is_contiguous(), "the host pool's tensor must be contiguous"
        self._base = host_pages.data_ptr()
        self._registered = host_pages.numel() * host_pages.element_size()
        rc = self._store.register_buffer(self._base, self._registered)
        if rc != 0:
            raise RuntimeError(f"Mooncake Store couldn't register the host pool ({self._registered} bytes): {rc}")
        self._layers = host_pages.shape[0]
        self._num_pages = host_pages.shape[1]
        self._block = host_pages[0, 0].numel() * host_pages.element_size()
        self._sizes = [self._block] * self._layers
        self._pages_per_call = max(1, _tcp_queue() // 4 // self._layers) if self._tcp else self._num_pages

    def _blocks(self, page: int) -> list[int]:
        return [self._base + (layer * self._num_pages + page) * self._block for layer in range(self._layers)]

    def exists(self, keys: list[str]) -> list[bool]:
        self._reopen_if_closed()
        with self._lock.shared():
            codes = self._store.batch_is_exist(keys)
        errors = [code for code in codes if code < 0]
        if errors:
            raise RuntimeError(f"Mooncake couldn't say whether {len(errors)} of {len(codes)} keys exist ({errors[0]})")
        return [code == 1 for code in codes]

    def put(self, keys: list[str], pages: list[int]) -> list[bool]:
        codes: list[int] = []
        self._reopen_if_closed()
        with self._lock.shared():
            for at in range(0, len(keys), self._pages_per_call):
                part = pages[at:at + self._pages_per_call]
                codes += self._store.batch_put_from_multi_buffers(
                    keys[at:at + self._pages_per_call], [self._blocks(page) for page in part],
                    [self._sizes] * len(part),
                )
        return [code == 0 for code in codes]

    def get(self, keys: list[str], pages: list[int]) -> list[GetResult]:
        """Mooncake stops waiting on a transfer that fails without stopping it, so its bytes can land in the
        host pages long after (seen on 0.3.13 over TCP), and only closing the client ends it. So after a
        failed page, the client is closed and a fresh one opened before this returns: a get that returns has
        stopped all its transfers. Raises if the client couldn't be closed or reopened."""
        self._reopen_if_closed()
        try:
            with self._lock.shared():
                results = self._get(keys, pages)
        except Exception:
            logger.warning("KV store: a Mooncake get raised; restarting the client", exc_info=True)
            results = [GetResult.FAILED] * len(keys)
        if GetResult.FAILED in results:
            self._restart()
        return results

    def _restart(self) -> None:
        """Close the client, once the calls in flight on the other store threads return, and open another."""
        with self._lock.alone():
            self._close_client()
            # never left pointing at the closed client: a call finds none and reopens it, or raises
            self._store = None
            self._store = self._open_registered()
        logger.warning("KV store: closed and reopened the Mooncake client after a failed read")

    def _reopen_if_closed(self) -> None:
        """Open a client a failed reopen left missing; raises while Mooncake can't be reached."""
        if self._store is None:
            with self._lock.alone():
                if self._store is None:
                    self._store = self._open_registered()

    def _open_registered(self):
        client = self._connect()
        if self._registered:
            rc = client.register_buffer(self._base, self._registered)
            if rc != 0:
                client.close()
                raise RuntimeError(f"Mooncake Store couldn't register the host pool again: {rc}")
        return client

    def _get(self, keys: list[str], pages: list[int]) -> list[GetResult]:
        whole = self._block * self._layers
        results: list[GetResult] = []
        for at in range(0, len(keys), self._pages_per_call):
            part = pages[at:at + self._pages_per_call]
            codes = self._store.batch_get_into_multi_buffers(
                keys[at:at + self._pages_per_call], [self._blocks(page) for page in part], [self._sizes] * len(part),
            )
            chunk = [
                GetResult.ARRIVED if code == whole else GetResult.MISSING if code in _NOT_THERE else GetResult.FAILED
                for code in codes
            ]
            results += chunk
            if any(result is not GetResult.ARRIVED for result in chunk):
                # only the arrived pages from the first are filed, so the rest isn't read: nothing sent, so missing
                return results + [GetResult.MISSING] * (len(keys) - len(results))
        return results

    def close(self) -> None:
        self._close_client()
        self._registered = 0

    def _close_client(self) -> None:
        if self._store is None:
            return
        if self._registered:
            self._store.unregister_buffer(self._base)
        close = getattr(self._store, "close", None)
        if close is not None:
            close()



class _SharedLock:
    """The store threads share the client; a restart takes it alone, once the calls in flight have returned.
    New calls wait while a restart waits, so it can't be starved."""

    def __init__(self):
        self._cond = threading.Condition()
        self._users = 0
        self._restarting = False

    @contextmanager
    def shared(self):
        with self._cond:
            self._cond.wait_for(lambda: not self._restarting)
            self._users += 1
        try:
            yield
        finally:
            with self._cond:
                self._users -= 1
                self._cond.notify_all()

    @contextmanager
    def alone(self):
        with self._cond:
            self._cond.wait_for(lambda: not self._restarting)
            self._restarting = True
            self._cond.wait_for(lambda: self._users == 0)
        try:
            yield
        finally:
            with self._cond:
                self._restarting = False
                self._cond.notify_all()


def _default_hostname(master: str) -> str:
    """This host's address on the interface that reaches the master, with a port of its own: every M*
    process on a machine is its own store client, and Mooncake names a client by this address."""
    host = master.rsplit(":", 1)[0] or "127.0.0.1"
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            probe.connect((host, 9))  # no packet is sent; this only picks the outgoing interface
            ip = probe.getsockname()[0]
        except OSError:
            ip = "127.0.0.1"
    with socket.socket() as free:
        free.bind((ip, 0))
        port = free.getsockname()[1]
    return f"{ip}:{port}"
