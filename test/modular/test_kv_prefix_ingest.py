"""A repeat ingest must not put back a prefix chain the stream dropped.

The conductor sends one NewRequest per partition, so a worker serving two
ingests a request twice, and the second can arrive after steps committed. A
stream drops its chain when a write stops matching the keys (another walk wrote
the keyed span, say). Seeding it again on the second ingest would index pages
under keys that don't describe them, and a later request with that prompt would
read the wrong KV.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources.base import KeyedWalks
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVReqConfig, KVStep, PagedKVConfig, PrefixSpan, StreamLayout
from mstar.engine.resources.kv.keys import chain
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.step import Segment, StepContext

PAGE_SIZE = 128
RID = "r0"
WALK = "prefill"


class _StubTransfer:
    def __init__(self, transfer_engine_info, kv_cache, **kwargs):
        del transfer_engine_info, kv_cache, kwargs

    def get_kv_transfer_info(self, **kwargs):
        del kwargs

    def remove_request(self, request_id):
        del request_id

    def cleanup(self):
        pass


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)


def _manager() -> KVManager:
    kv = KVManager(
        cfg=PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=8192,
            max_num_pages=64, page_size=PAGE_SIZE,
        ),
        name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    kv.enable_prefix_cache(b"a root", walks={"main": KeyedWalks(writes=(WALK,))})
    return kv


def _config(n_tokens: int) -> KVReqConfig:
    tokens = list(range(n_tokens))
    keys = chain([tokens[at:at + PAGE_SIZE] for at in range(0, n_tokens, PAGE_SIZE)])
    return KVReqConfig(
        prefix_keys={"main": keys},
        kv_layout={"main": StreamLayout(spans=(PrefixSpan(n_tokens, n_tokens, WALK),), keyed_spans=1)},
    )


def _step(kv: KVManager, span: int, walk: str) -> None:
    step = KVStep(segments=(Segment(RID, "main", span),))
    ctx = StepContext(request_ids=(RID,), graph_walk=walk, slot=0, capture=False)
    assert kv.admit(step, ctx).ok
    kv.plan(step, ctx)
    kv.commit(step, ctx)


def test_a_repeat_ingest_keeps_a_dropped_chain_dropped():
    kv = _manager()
    config = _config(2 * PAGE_SIZE)
    kv.ingest_request(RID, config)
    # a walk the keys don't name writes the first page, so the chain ends
    _step(kv, PAGE_SIZE, "image")
    assert kv._streams[RID]["main"].chain is None

    # the second partition's NewRequest
    kv.ingest_request(RID, config)
    assert kv._streams[RID]["main"].chain is None
    _step(kv, PAGE_SIZE, WALK)

    assert kv._streams[RID]["main"].chain is None
    assert kv._index.pages() == [], "pages were indexed under keys that don't describe them"
