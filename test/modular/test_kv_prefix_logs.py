"""What a deployment needs to read off a running cache.

Three numbers decide which of the follow-ups is worth building, and none of them
can be recovered afterwards: how much of each prompt was already here, whether
whole inputs recur (which is what would make skipping an encoder worth it), and
which copy of the node each request landed on (which is what would make routing
by prefix worth it). They are per request, so they are logged where a request is
admitted, once, however many times it is admitted.
"""

from __future__ import annotations

import logging
import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVConfig, KVReqConfig, KVStep
from mstar.engine.resources.kv.keys import chain
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.engine.resources.step import Segment, StepContext

PAGE_SIZE = 16
ROOT = b"a root"
NODE = "LLM"
WALK = "prefill"
REPLICA = "worker-3"


class _StubTransfer:
    """No engine, no bytes moved."""

    def __init__(self, transfer_engine_info, kv_cache):
        del transfer_engine_info, kv_cache

    def get_kv_transfer_info(self):
        return None

    def start_async_retrieve(self, **kwargs):
        del kwargs

    def cleanup(self):
        pass


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)


def _manager(prefix_cache: bool = True) -> KVManager:
    kv = KVManager(
        cfg=KVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=4096,
            max_num_pages=64, page_size=PAGE_SIZE, prefix_cache=prefix_cache,
        ),
        name="kv", joint_comm_group=None,
        transfer_engine_info=TransferEngineInfo(
            my_entity_id=REPLICA, my_session_id="s", transfer_engine=None,
        ),
        device=torch.device("cpu"), dtype=torch.float32,
    )
    kv.enable_prefix_cache(ROOT)
    return kv


def _ingest(kv: KVManager, rid: str, tokens: list[int], keyed: bool = True) -> None:
    if not keyed:
        kv.ingest_request(rid, KVReqConfig())
        return
    whole = len(tokens) // PAGE_SIZE
    kv.ingest_request(rid, KVReqConfig(
        prefix_keys={"main": chain([
            tokens[at:at + PAGE_SIZE] for at in range(0, len(tokens), PAGE_SIZE)
        ])},
        prefix_tail={"main": tokens[whole * PAGE_SIZE:]},
    ))


def _run(kv: KVManager, rid: str, span: int):
    step = KVStep(segments=(Segment(rid, "main", span),))
    ctx = StepContext(
        request_ids=(rid,), graph_walk=WALK, slot=0, capture=False,
    )
    outcome = kv.admit(step, ctx)
    if outcome.ok:
        kv.plan(step, ctx)
        kv.commit(step, ctx)
    return outcome


def _lines(caplog) -> list[str]:
    return [
        record.getMessage() for record in caplog.records
        if "matched" in record.getMessage()
    ]


# ── what the line says ──────────────────────────────────────────────────


def test_a_miss_is_reported_as_a_miss(caplog):
    kv = _manager()
    tokens = list(range(64))
    _ingest(kv, "r0", tokens)

    with caplog.at_level(logging.INFO, logger=manager_mod.__name__):
        _run(kv, "r0", len(tokens))

    assert len(_lines(caplog)) == 1
    line = _lines(caplog)[0]
    assert "matched 0 of its 4 pages" in line
    assert "whole prompt already cached False" in line
    assert f"replica {REPLICA}" in line


def test_a_hit_reports_how_much_of_the_prompt_was_already_here(caplog):
    kv = _manager()
    tokens = list(range(64))
    _ingest(kv, "a", tokens)
    _run(kv, "a", len(tokens))
    kv.remove_request("a")
    _ingest(kv, "b", tokens)
    kv.resolve_cached_prefix("b", NODE, WALK)

    with caplog.at_level(logging.INFO, logger=manager_mod.__name__):
        _run(kv, "b", len(tokens))

    line = _lines(caplog)[0]
    assert "matched 3 of its 4 pages" in line, line
    assert "whole prompt already cached False" in line, (
        "three of four pages is not the whole prompt"
    )


def test_a_whole_prompt_that_came_back_says_so(caplog):
    kv = _manager()
    tokens = list(range(64))
    _ingest(kv, "a", tokens)
    _run(kv, "a", len(tokens))
    kv.remove_request("a")
    # the same prompt with a few tokens more: those do not finish a page, so
    # every whole page this request has is one the first request left behind
    longer = tokens + list(range(9000, 9004))
    _ingest(kv, "b", longer)
    kv.resolve_cached_prefix("b", NODE, WALK)

    with caplog.at_level(logging.INFO, logger=manager_mod.__name__):
        _run(kv, "b", len(longer))

    assert "whole prompt already cached True" in _lines(caplog)[0], _lines(caplog)[0]


# ── who is reported, and how often ──────────────────────────────────────


def test_an_undeclared_request_is_not_reported(caplog):
    kv = _manager()
    _ingest(kv, "r0", list(range(64)), keyed=False)

    with caplog.at_level(logging.INFO, logger=manager_mod.__name__):
        _run(kv, "r0", 64)

    assert _lines(caplog) == [], "a request nobody keyed was reported anyway"


def test_a_deployment_that_turned_the_cache_off_reports_nothing(caplog):
    kv = _manager(prefix_cache=False)
    _ingest(kv, "r0", list(range(64)))

    with caplog.at_level(logging.INFO, logger=manager_mod.__name__):
        _run(kv, "r0", 64)

    assert _lines(caplog) == [], (
        "a closed cache logged a miss for a request that could never hit"
    )


def test_a_request_admitted_twice_is_still_one_line(caplog):
    kv = _manager()
    tokens = list(range(64))
    _ingest(kv, "r0", tokens)

    with caplog.at_level(logging.INFO, logger=manager_mod.__name__):
        _run(kv, "r0", len(tokens))
        _run(kv, "r0", PAGE_SIZE)

    assert len(_lines(caplog)) == 1, (
        "every step of a request would be a line, not every request"
    )
