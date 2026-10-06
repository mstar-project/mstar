"""What it costs a pool that backfills to be asked about a long queue.

The scheduler asks readiness of every request still waiting, on every pass, so
what one ask costs is paid once per waiting request per pass.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch
from test_kv_admission_peak import (
    DECODE,
    PAGE_SIZE,
    PREFILL,
    _ingest,
    _request,
    _StubTransfer,
)

from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import PagedKVConfig
from mstar.engine.resources.kv.manager import KVManager


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)
    for name in (
        "MSTAR_KV_ADMISSION_FIT", "MSTAR_KV_ADMISSION_ORDER", "MSTAR_KV_BACKFILL_WINDOW",
        "MSTAR_STEP_TELEMETRY_DIR",
    ):
        monkeypatch.delenv(name, raising=False)


def _manager(
    max_num_pages: int = 16, fit: str = "peak", order: str = "backfill", **config,
) -> KVManager:
    kv = KVManager(
        cfg=PagedKVConfig(
            num_layers=1, num_kv_heads=2, head_dim=8, max_seq_len=4096,
            max_num_pages=max_num_pages, page_size=PAGE_SIZE,
            admission_fit=fit, admission_order=order, **config,
        ),
        name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    kv.enable_prefix_cache(b"a root", {"main": (PREFILL, DECODE)})
    return kv


# ── the padding rows ────────────────────────────────────────────────────


def test_what_the_padding_rows_keep_is_counted_out_of_the_capacity(monkeypatch):
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)
    kv = _manager(32, fit="sum", order="fifo")
    assert kv._capacity() == 31

    kv.ingest_request(-1, _request(list(range(100)), 4))
    assert kv._alloc(-1, "main", 3 * PAGE_SIZE).success
    _ingest(kv, "real", 40, 4)
    assert kv._alloc("real", "main", 2 * PAGE_SIZE).success
    assert kv._capacity() == 31 - 3, "the padding row's pages are not the pool's to promise"

    kv.remove_request(-1)
    assert kv._capacity() == 31
