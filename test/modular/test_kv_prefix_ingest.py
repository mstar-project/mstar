"""A second partition's NewRequest must not put back a prefix chain the stream dropped.

The conductor sends one NewRequest per partition, so a worker serving two gets
the request twice, and the second can arrive after steps committed. A stream
drops its chain when a write stops matching the keys (another walk wrote the
keyed span, say). Seeding it again would index pages under keys that don't
describe them, and a later request with that prompt would read the wrong KV.
The worker opens the engine's state on the first NewRequest only.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.resources import StepRunner
from mstar.engine.resources.base import KeyedWalks
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVReqConfig, KVStep, PagedKVConfig, PrefixSpan, StreamLayout
from mstar.engine.resources.kv.keys import chain
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.step import Segment, StepContext
from mstar.utils.ipc_format import NewRequest
from mstar.worker.node_manager_utils import RequestStateManager
from mstar.worker.rid_table import RidTable
from mstar.worker.stream_buffers import StreamBufferManager
from mstar.worker.worker import Worker

PAGE_SIZE = 16
KV = "kv"
NODE = "LLM"
WALK = "prefill"
WIRE = "req-1"


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
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=4096,
            max_num_pages=64, page_size=PAGE_SIZE,
        ),
        name=KV, joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    kv.enable_prefix_cache(b"a root", walks={"main": KeyedWalks(writes=(WALK,))})
    return kv


def _worker(runner: StepRunner) -> Worker:
    """The real ``_add_new_request``, with the engine cut down to its resources."""
    rids = RidTable()
    w = Worker.__new__(Worker)
    w.worker_id = "w0"
    w._draining_rids, w._reads_done_sent, w._unprocessed_messages = set(), set(), {}
    w._graph_runtime = SimpleNamespace(
        add_request=lambda request_id, **kw: rids.intern(request_id),
        get_rid_handle=rids.handle,
        get_sharding_config=lambda rid: None,
        register_stop_limits=lambda rid, info: None,
    )
    w.request_state = RequestStateManager()
    w.scheduler = SimpleNamespace(note_arrival=lambda rid: None)
    w.tensor_manager = SimpleNamespace(
        register_request=lambda rid, sharding: None,
        start_read_tensors=lambda rid, edges, graph_walk=None: [],
    )
    w.engine_manager = SimpleNamespace(
        engine=SimpleNamespace(publisher_nodes={}),
        add_request=lambda rid, configs, info: runner.ingest_request(rid, configs),
    )
    w.streams = StreamBufferManager(w.request_state, w.tensor_manager, [])
    w.wakeup_event = SimpleNamespace(register_futures=lambda futures: None)
    return w


def _new_request(partition: str, n_tokens: int) -> NewRequest:
    """One partition's NewRequest, each with its own copy of the configs as off the wire."""
    tokens = list(range(n_tokens))
    config = KVReqConfig(
        prefix_keys={"main": chain([tokens[at:at + PAGE_SIZE] for at in range(0, n_tokens, PAGE_SIZE)])},
        kv_layout={"main": StreamLayout(spans=(PrefixSpan(n_tokens, n_tokens, WALK),), keyed_spans=1)},
    )
    return NewRequest(
        request_id=WIRE,
        request_info=CurrentForwardPassInfo(
            request_id=WIRE, graph_walk=WALK, fwd_index=0, random_seed=0, max_tokens=1,
            partition_name=partition, resource_configs={KV: config},
        ),
        partition_worker_graph_ids=[], worker_graph_to_workers={}, initial_inputs=[], loop_caps={},
    )


def _step(kv: KVManager, rid: int, span: int, walk: str) -> None:
    step = KVStep(segments=(Segment(rid, "main", span),))
    ctx = StepContext(request_ids=(rid,), graph_walk=walk, slot=0, capture=False)
    assert kv.admit(step, ctx).ok
    kv.plan(step, ctx)
    kv.commit(step, ctx)


def test_a_second_partition_keeps_a_dropped_chain_dropped():
    kv = _manager()
    w = _worker(StepRunner({KV: kv}, node_resources={NODE: [KV]}))
    w._add_new_request(_new_request("A", 2 * PAGE_SIZE))
    rid = w._rid(WIRE)
    # a walk the keys don't name writes the first page, so the chain ends
    _step(kv, rid, PAGE_SIZE, "image")
    assert kv._streams[rid]["main"].chain is None

    w._add_new_request(_new_request("B", 2 * PAGE_SIZE))
    _step(kv, rid, PAGE_SIZE, WALK)

    assert kv._streams[rid]["main"].chain is None
    assert kv._index.pages() == [], "pages were indexed under keys that don't describe them"
