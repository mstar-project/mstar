"""What a deployment needs to read off a running cache.

Three numbers decide which of the follow-ups is worth building, and none of them
can be recovered afterwards: how much of each prompt was already here, whether
whole inputs recur (which is what would make skipping an encoder worth it), and
which copy of the node each request landed on (which is what would make routing
by prefix worth it). They are per request, so they are logged where a request is
admitted, once, however many times it is admitted.

A request that reaches a declared node with no keys is never reported, so a
break between the preprocess worker and the node leaves an empty cache and a
quiet log. The node says so once, not once a request.
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

    def __init__(self, transfer_engine_info, kv_cache, **kwargs):
        del transfer_engine_info, kv_cache, kwargs

    def get_kv_transfer_info(self, **kwargs):
        del kwargs

    def owns_transfer_info(self, transfer_info, **kwargs):
        del kwargs
        return transfer_info == self.get_kv_transfer_info()

    def remove_request(self, request_id):
        del request_id

    def start_async_retrieve(self, **kwargs):
        del kwargs

    def cleanup(self):
        pass


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)


def _manager(prefix_cache: bool = True, walks=None) -> KVManager:
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
    kv.enable_prefix_cache(ROOT, walks)
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

    assert len(_lines(caplog)) == 1, "a declared request was not reported once"
    line = _lines(caplog)[0]
    assert "matched 0 of its 4 pages" in line, line
    assert "whole prompt already cached False" in line, line
    assert f"replica {REPLICA}" in line, (
        f"the line names no replica, so nothing says which copy served it: {line}"
    )


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
    _ingest(kv, "r1", list(range(64)))

    with caplog.at_level(logging.INFO, logger=manager_mod.__name__):
        _run(kv, "r0", 64)
        undeclared = _lines(caplog)
        _run(kv, "r1", 64)

    assert undeclared == [], "a request nobody keyed was reported anyway"
    assert len(_lines(caplog)) == 1, "the keyed request beside it was not reported"


def test_a_deployment_that_turned_the_cache_off_reports_nothing(caplog):
    closed = _manager(prefix_cache=False)
    _ingest(closed, "r0", list(range(64)))
    open_cache = _manager()
    _ingest(open_cache, "r0", list(range(64)))

    with caplog.at_level(logging.INFO, logger=manager_mod.__name__):
        _run(closed, "r0", 64)
        shut = _lines(caplog)
        _run(open_cache, "r0", 64)

    assert shut == [], (
        "a closed cache logged a miss for a request that could never hit"
    )
    assert len(_lines(caplog)) == 1, "the same request on an open cache was silent too"


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


# ── a declared node that its keys never reach ───────────────────────────

DECLARED = {"main": (WALK, "decode")}


def _serve(kv: KVManager, count: int, keyed: bool) -> None:
    for n in range(count):
        rid = f"r{n}"
        _ingest(kv, rid, list(range(64)), keyed=keyed)
        kv.resolve_cached_prefix(rid, NODE, WALK)
        _run(kv, rid, 64)
        kv.remove_request(rid)


def _warnings(caplog) -> list[str]:
    return [
        record.getMessage() for record in caplog.records
        if record.levelno == logging.WARNING
    ]


def test_a_declared_node_its_keys_never_reach_warns_once(caplog):
    kv = _manager(walks=DECLARED)

    with caplog.at_level(logging.INFO, logger=manager_mod.__name__):
        _serve(kv, 10, keyed=False)

    warnings = _warnings(caplog)
    assert len(warnings) == 1, f"ten unkeyed requests gave {len(warnings)} warnings, not one"
    assert NODE in warnings[0] and "main" in warnings[0], (
        f"the warning does not name the node and the label that went unkeyed: {warnings[0]}"
    )


def test_a_declared_node_that_receives_its_keys_never_warns(caplog):
    kv = _manager(walks=DECLARED)

    with caplog.at_level(logging.INFO, logger=manager_mod.__name__):
        _serve(kv, 10, keyed=True)

    assert _warnings(caplog) == [], "a node receiving its keys was warned it had none"


def test_an_undeclared_node_never_warns(caplog):
    kv = _manager()

    with caplog.at_level(logging.INFO, logger=manager_mod.__name__):
        _serve(kv, 10, keyed=False)

    assert _warnings(caplog) == [], (
        "a node the model never declared was warned about keys it was never owed"
    )
