"""A request never retrieves its KV from itself.

The conductor sends one ``NewRequest`` per (worker, partition), all naming the
same request id, so a worker serving two partitions ingests the same rid twice.
Re-ingesting used to replace the stream dict — resetting ``stored_len`` under a
node that had already filled it — after which the request's publish info named
more tokens than the stream held and ``admit_retrieve`` fired a transfer from
this engine's own CUDA IPC handle. Opening your own handle raises
``CUDA error: invalid device context``.
"""

from __future__ import annotations

import sys
import threading

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVConfig, KVReqConfig, KVStep
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.step import Segment, StepContext

OWN_HANDLE = "own-ipc-handle"
PEER_HANDLE = "peer-ipc-handle"


class _StubTransfer:
    """Records retrieves instead of touching CUDA."""

    started: list[dict] = []
    published: list[dict] = []
    remove_started: threading.Event | None = None
    allow_remove: threading.Event | None = None

    def __init__(self, transfer_engine_info, kv_cache, **kwargs):
        del transfer_engine_info, kv_cache, kwargs

    def get_kv_transfer_info(self, **kwargs):
        type(self).published.append(kwargs)
        return OWN_HANDLE

    def start_async_retrieve(self, **kwargs):
        type(self).started.append(kwargs)

    def owns_transfer_info(self, transfer_info, **kwargs):
        del kwargs
        return transfer_info == OWN_HANDLE

    def cleanup(self):
        pass

    def remove_request(self, request_id):
        del request_id
        if self.remove_started is not None:
            self.remove_started.set()
        if self.allow_remove is not None:
            assert self.allow_remove.wait(timeout=5)


@pytest.fixture(autouse=True)
def _stub(monkeypatch):
    _StubTransfer.started = []
    _StubTransfer.published = []
    _StubTransfer.remove_started = None
    _StubTransfer.allow_remove = None
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)


def _manager(max_num_pages=64, page_size=16) -> KVManager:
    return KVManager(
        cfg=KVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=4096,
            max_num_pages=max_num_pages, page_size=page_size,
        ),
        name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
    )


def _ctx(*rids):
    return StepContext(
        request_ids=tuple(rids), graph_walk="w", slot=0, capture=False,
    )


def _grow(kv, rid, span, label="main"):
    step = KVStep(segments=(Segment(rid, label, span),))
    ctx = _ctx(rid)
    assert kv.admit(step, ctx).ok
    kv.commit(step, ctx)


# ── the fix ─────────────────────────────────────────────────────────────


def test_re_ingesting_a_request_keeps_what_it_already_stored():
    """The regression: partition B's NewRequest wiped partition A's stream."""
    kv = _manager()
    kv.ingest_request("r0")
    _grow(kv, "r0", 100)

    kv.ingest_request("r0")  # second partition, same rid

    assert kv._streams["r0"]["main"].stored_len == 100


def test_re_ingesting_keeps_the_pages_too():
    kv = _manager()
    kv.ingest_request("r0")
    _grow(kv, "r0", 100)
    pages = list(kv._streams["r0"]["main"].page_indices)
    free = kv._arena.num_free

    kv.ingest_request("r0")

    assert kv._streams["r0"]["main"].page_indices == pages
    assert kv._arena.num_free == free, "a wiped stream would orphan its pages"


def test_a_fresh_request_still_starts_empty():
    kv = _manager()
    kv.ingest_request("r0")
    _grow(kv, "r0", 100)
    kv.remove_request("r0")

    kv.ingest_request("r0")  # id reused after removal

    assert kv._streams["r0"]["main"].stored_len == 0


# ── defence in depth ────────────────────────────────────────────────────


def test_our_own_publish_never_starts_a_retrieve():
    """Even with the lengths disagreeing, reading our own handle is a CUDA
    error — there is nothing to move from ourselves."""
    kv = _manager()
    kv.ingest_request("r0")
    _grow(kv, "r0", 100)
    published = kv.publish("r0")
    # force the disagreement the old ingest produced
    kv._streams["r0"]["main"].stored_len = 0

    out = kv.admit_retrieve("r0", "node", "w", published)

    assert out.ok
    assert _StubTransfer.started == [], "must not read from our own cache"


def test_a_peer_publish_still_retrieves():
    """The guard must not disable real disaggregated transfers."""
    kv = _manager()
    kv.ingest_request("r0")
    published = kv.publish("r0")
    seq = published.get(0)["main"]
    published.info[0]["main"] = type(seq)(
        seq_len=100, latest_kv_transfer_info=PEER_HANDLE, page_indices=[1, 2, 3],
    )

    out = kv.admit_retrieve("r0", "node", "w", published)

    assert out.ok
    assert len(_StubTransfer.started) == 1
    assert _StubTransfer.started[0]["kv_transfer_info"] == PEER_HANDLE


# ── publish is a snapshot, taken under the lock ─────────────────────────


def test_publish_copies_the_page_list():
    """A consumer holds these while this engine keeps allocating; sharing the
    live list would hand it pages that moved."""
    kv = _manager()
    kv.ingest_request("r0")
    _grow(kv, "r0", 100)
    published = kv.publish("r0")
    pages = list(published.get(0)["main"].page_indices)

    _grow(kv, "r0", 200)  # extends the stream's own list

    assert published.get(0)["main"].page_indices == pages


def test_publish_exports_only_labels_declared_for_remote_consumers():
    kv = _manager()
    kv.ingest_request(
        "r0",
        KVReqConfig(
            publish_labels_per_node_walk={
                ("producer", "prefill"): ["branch"],
            },
            final_publish_labels_per_node_walk={
                ("producer", "decode"): ["branch"],
            },
        ),
    )
    _grow(kv, "r0", 16)
    _grow(kv, "r0", 16, label="branch")

    assert kv.publish("r0", "producer", "decode") is None
    assert kv.publish_after_stop("r0", "producer", "prefill") is None
    published = kv.publish("r0", "producer", "prefill")
    final_published = kv.publish_after_stop("r0", "producer", "decode")

    assert set(published.get(0)) == {"branch"}
    assert set(final_published.get(0)) == {"branch"}


def test_decode_exports_kv_once_after_stop():
    kv = _manager()
    kv.ingest_request(
        "r0",
        KVReqConfig(
            publish_labels_per_node_walk={},
            final_publish_labels_per_node_walk={
                ("LLM", "decode"): ["main"],
            },
        ),
    )

    for _ in range(4):
        _grow(kv, "r0", 16)
        assert kv.publish("r0", "LLM", "decode") is None

    assert _StubTransfer.published == []

    published = kv.publish_after_stop("r0", "LLM", "decode")

    assert set(published.get(0)) == {"main"}
    assert len(_StubTransfer.published) == 1
    assert _StubTransfer.published[0]["seq_len"] == 64


def test_publish_is_consistent_against_a_concurrent_commit():
    """publish runs on the GPU thread while the scheduler thread commits; a
    published length must never name pages the snapshot doesn't include."""
    kv = _manager(max_num_pages=512, page_size=16)
    kv.ingest_request("r0")
    _grow(kv, "r0", 16)

    stop = threading.Event()
    torn: list[tuple[int, int]] = []

    def _committer():
        # stops on its own once the arena runs dry, so the thread never raises
        while not stop.is_set():
            step = KVStep(segments=(Segment("r0", "main", 16),))
            ctx = _ctx("r0")
            if not kv.admit(step, ctx).ok:
                return
            kv.commit(step, ctx)

    t = threading.Thread(target=_committer, daemon=True)
    t.start()
    try:
        for _ in range(300):
            info = kv.publish("r0").get(0)["main"]
            have = len(info.page_indices) * 16
            if info.seq_len > have:
                torn.append((info.seq_len, have))
    finally:
        stop.set()
        t.join(timeout=5)

    assert not torn, f"published a length past its own pages: {torn[:5]}"


def test_remove_keeps_transfer_cleanup_atomic_with_publish():
    kv = _manager()
    kv.ingest_request("r0")
    _grow(kv, "r0", 16)
    _StubTransfer.remove_started = threading.Event()
    _StubTransfer.allow_remove = threading.Event()

    remover = threading.Thread(target=kv.remove_request, args=("r0",))
    remover.start()
    assert _StubTransfer.remove_started.wait(timeout=5)

    result = []
    publish_done = threading.Event()

    def _publish():
        result.append(kv.publish("r0"))
        publish_done.set()

    publisher = threading.Thread(target=_publish)
    publisher.start()
    assert not publish_done.wait(timeout=0.05)

    _StubTransfer.allow_remove.set()
    remover.join(timeout=5)
    publisher.join(timeout=5)

    assert result == [None]
