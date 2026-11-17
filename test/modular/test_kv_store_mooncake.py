"""The Mooncake backend against a real mooncake_master on this machine, over TCP, on CPU memory.

Skips where Mooncake doesn't load (its default wheel needs the CUDA 12 runtime) or the master binary is
missing.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import torch

pytest.importorskip("mooncake.store", exc_type=ImportError)

from mstar.engine.resources.kv.store import MooncakeStoreConfig, page_layout, store_key  # noqa: E402
from mstar.engine.resources.kv.store.base import GetResult  # noqa: E402
from mstar.engine.resources.kv.store.mooncake import MooncakeStore  # noqa: E402

LAYERS, PAGES, BLOCK = 3, 8, 4096


def _master_binary() -> str | None:
    beside = Path(sys.executable).with_name("mooncake_master")
    return str(beside) if beside.exists() else shutil.which("mooncake_master")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def _running_master():
    """A mooncake_master on free ports, as (its address, its HTTP metadata URL)."""
    binary = _master_binary()
    if binary is None:
        pytest.skip("mooncake_master isn't installed")
    rpc, http = _free_port(), _free_port()
    proc = subprocess.Popen(
        [binary, f"--rpc_port={rpc}", f"--metrics_port={_free_port()}", "--enable_http_metadata_server=true",
         "--http_metadata_server_host=127.0.0.1", f"--http_metadata_server_port={http}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", rpc)) == 0:
                break
        time.sleep(0.1)
    else:
        proc.kill()
        pytest.skip("mooncake_master didn't start")
    try:
        yield f"127.0.0.1:{rpc}", f"http://127.0.0.1:{http}/metadata"
    finally:
        proc.terminate()
        proc.wait(10)


@pytest.fixture(scope="module")
def master():
    with _running_master() as running:
        yield running


def test_pages_round_trip_through_mooncake(master):
    address, metadata = master
    config = MooncakeStoreConfig(
        master=address, metadata=metadata,
        local_hostname=f"127.0.0.1:{_free_port()}", segment_bytes=64 << 20,
    )
    store = MooncakeStore(config)
    host = torch.zeros((LAYERS, PAGES, BLOCK // 4), dtype=torch.float32)
    store.bind(host)
    try:
        layout = page_layout(host, "kv")
        keys = [store_key(layout, bytes([i]) * 32) for i in range(3)]
        for i, page in enumerate((1, 4, 6)):
            host[:, page] = torch.arange(LAYERS).view(-1, 1) * 10 + i
        sent = host[:, [1, 4, 6]].clone()

        assert store.exists(keys) == [False, False, False]
        assert store.put(keys, [1, 4, 6]) == [True, True, True]
        assert store.put(keys[:1], [1]) == [True], "a put of a key the store already has didn't count as stored"
        assert store.exists(keys + [store_key(layout, b"\xff" * 32)]) == [True, True, True, False]

        host.zero_()
        assert store.get(keys, [2, 3, 7]) == [GetResult.ARRIVED] * 3
        assert torch.equal(host[:, [2, 3, 7]], sent), "pages came back from Mooncake with different bytes"
        assert store.get([store_key(layout, b"\xee" * 32)], [0]) == [GetResult.MISSING], "a missing key read as found"
    finally:
        store.close()


def test_two_clients_on_one_machine_share_pages(master):
    """Two clients with the default address, as two M* processes on one machine get: one holds the store's
    memory, and the other, which gives none, reads what the first wrote and writes into it."""
    address, metadata = master
    holder = MooncakeStore(MooncakeStoreConfig(master=address, metadata=metadata, segment_bytes=64 << 20))
    reader = MooncakeStore(MooncakeStoreConfig(master=address, metadata=metadata))
    a = torch.zeros((LAYERS, PAGES, BLOCK // 4))
    b = torch.zeros((LAYERS, PAGES, BLOCK // 4))
    holder.bind(a)
    reader.bind(b)
    try:
        layout = page_layout(a, "kv")
        mine, theirs = store_key(layout, b"\x01" * 32), store_key(layout, b"\x02" * 32)
        a[:, 0] = 7.0
        b[:, 5] = 3.0
        assert holder.put([mine], [0]) == [True]
        assert reader.put([theirs], [5]) == [True], "a client that gives no memory couldn't write"

        assert reader.get([mine], [2]) == [GetResult.ARRIVED] and torch.equal(b[:, 2], a[:, 0])
        assert holder.get([theirs], [6]) == [GetResult.ARRIVED] and torch.equal(a[:, 6], b[:, 5])
    finally:
        reader.close()
        holder.close()


def test_a_read_past_mooncakes_tcp_queue_arrives(master):
    """A page is a transfer per layer, and Mooncake's TCP transport queues 2048 transfers a peer: one call for
    this read (32 layers of 80 pages) failed after a 60 s stall, so the backend splits it."""
    address, metadata = master
    holder = MooncakeStore(MooncakeStoreConfig(master=address, metadata=metadata, segment_bytes=64 << 20))
    reader = MooncakeStore(MooncakeStoreConfig(master=address, metadata=metadata))
    a = torch.randn((32, 80, 256))
    b = torch.zeros_like(a)
    holder.bind(a)
    reader.bind(b)
    try:
        keys = [store_key(page_layout(a, "kv"), page.to_bytes(32, "big")) for page in range(80)]
        assert all(holder.put(keys, list(range(80))))
        assert reader.get(keys, list(range(80))) == [GetResult.ARRIVED] * 80
        assert torch.equal(b, a), "pages came back from Mooncake with different bytes"
    finally:
        reader.close()
        holder.close()


def test_two_clients_putting_the_same_pages_at_once_both_succeed(master):
    """Replicas given one system prompt write the same keys; a put that loses the race must not count as refused,
    or five in a row would stop that process's writes."""
    address, metadata = master
    holder = MooncakeStore(MooncakeStoreConfig(master=address, metadata=metadata, segment_bytes=64 << 20))
    writers = [MooncakeStore(MooncakeStoreConfig(master=address, metadata=metadata)) for _ in range(2)]
    pools = [torch.full((LAYERS, PAGES, BLOCK // 4), float(i + 1)) for i in range(2)]
    for writer, pool in zip(writers, pools, strict=True):
        writer.bind(pool)
    holder.bind(torch.zeros((LAYERS, PAGES, BLOCK // 4)))
    try:
        keys = [store_key(page_layout(pools[0], "kv"), bytes([0xA0 + page]) * 32) for page in range(PAGES)]
        start = threading.Barrier(2)
        results: dict[int, list[bool]] = {}

        def put(i: int) -> None:
            start.wait()
            results[i] = writers[i].put(keys, list(range(PAGES)))

        threads = [threading.Thread(target=put, args=(i,)) for i in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30)
        assert results == {0: [True] * PAGES, 1: [True] * PAGES}, results
        assert holder.exists(keys) == [True] * PAGES
    finally:
        for client in (*writers, holder):
            client.close()


_HOLDER = """
import sys, time
from mooncake.store import MooncakeDistributedStore
store = MooncakeDistributedStore()
assert store.setup(sys.argv[2], "P2PHANDSHAKE", 64 << 20, 16 << 20, "tcp", "", sys.argv[1]) == 0
print("up", flush=True)
time.sleep(3600)
"""


def test_a_failed_read_stops_its_transfers_before_it_returns():
    """A read from a store process that stops answering fails after Mooncake's 60 s wait, but Mooncake keeps its
    transfers going, and they write into the host pages once the process answers again. The backend closes its
    client before the get returns, so nothing lands afterwards. Takes about 60 s: Mooncake's wait is fixed."""
    # a master of its own: the frozen, then killed, holder stays registered for a while and would take puts
    with _running_master() as (address, _):
        _read_through_a_frozen_holder(address)


def _read_through_a_frozen_holder(address: str) -> None:
    holder = subprocess.Popen(
        [sys.executable, "-c", _HOLDER, address, f"127.0.0.1:{_free_port()}"],
        stdout=subprocess.PIPE, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    try:
        assert holder.stdout.readline().strip() == "up"
        reader = MooncakeStore(MooncakeStoreConfig(master=address))
        host = torch.randn((LAYERS, PAGES, 16384))
        reader.bind(host)
        keys = [store_key(page_layout(host, "kv"), bytes([0xC0 + page]) * 32) for page in range(PAGES)]
        assert all(reader.put(keys, list(range(PAGES))))
        assert reader.get(keys[:1], [0]) == [GetResult.ARRIVED]  # the connection to the holder is up

        host.zero_()
        os.kill(holder.pid, signal.SIGSTOP)
        assert GetResult.FAILED in reader.get(keys, list(range(PAGES)))
        at_return = host.clone()
        os.kill(holder.pid, signal.SIGCONT)
        time.sleep(2)
        assert torch.equal(host, at_return), "a failed read's transfers wrote into the host pages after it returned"

        # the master dropped the frozen process's memory (its clients time out after 10 s), so ask, not read
        assert len(reader.exists(keys)) == PAGES, "the reopened client can't reach the master"
        reader.close()
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(holder.pid, signal.SIGCONT)
        holder.kill()
        holder.wait(10)


def test_one_client_serves_three_threads_at_once(master):
    """The KV store calls one client from its lookup, read and put threads at the same time."""
    address, metadata = master
    store = MooncakeStore(MooncakeStoreConfig(master=address, metadata=metadata, segment_bytes=64 << 20))
    host = torch.zeros((LAYERS, PAGES, BLOCK // 4))
    store.bind(host)
    layout = page_layout(host, "kv")
    base = [store_key(layout, bytes([200 + i]) * 32) for i in range(2)]
    host[:, 0], host[:, 1] = 1.0, 2.0
    assert store.put(base, [0, 1]) == [True, True]
    errors: list[str] = []

    def lookups():
        for _ in range(200):
            if store.exists(base) != [True, True]:
                errors.append("exists")

    def reads():
        for _ in range(200):
            if store.get(base, [2, 3]) != [GetResult.ARRIVED] * 2:
                errors.append("get")

    def puts():
        for i in range(200):
            if store.put([store_key(layout, i.to_bytes(32, "little"))], [4 + i % 4]) != [True]:
                errors.append("put")

    threads = [threading.Thread(target=run) for run in (lookups, reads, puts)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(60)
        assert errors == [], f"calls from three threads at once failed: {sorted(set(errors))}"
        assert torch.equal(host[:, 2], host[:, 0]) and torch.equal(host[:, 3], host[:, 1])
    finally:
        store.close()


def test_the_default_p2p_handshake_needs_only_the_master(master):
    """``metadata: P2PHANDSHAKE``, the default, finds peers without the HTTP metadata server."""
    address, _ = master
    holder = MooncakeStore(MooncakeStoreConfig(master=address, segment_bytes=64 << 20))
    reader = MooncakeStore(MooncakeStoreConfig(master=address))
    a = torch.zeros((LAYERS, PAGES, BLOCK // 4))
    b = torch.zeros((LAYERS, PAGES, BLOCK // 4))
    holder.bind(a)
    reader.bind(b)
    try:
        key = store_key(page_layout(a, "kv"), b"\x06" * 32)
        a[:, 0] = 4.0
        assert holder.put([key], [0]) == [True]
        assert reader.exists([key]) == [True]
        assert reader.get([key], [1]) == [GetResult.ARRIVED] and torch.equal(b[:, 1], a[:, 0])
    finally:
        reader.close()
        holder.close()
