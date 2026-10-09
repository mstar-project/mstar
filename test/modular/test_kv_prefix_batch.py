"""The batched prefix-chain extension (one call per step off the stop check's
host rows) keys exactly what the per-request form keys."""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, ".")

from mstar.engine.resources.runner import StepRunner  # noqa: E402
from mstar.model.submodule_base import HostRows  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "kv_prefix_decode_helpers", pathlib.Path(__file__).with_name("test_kv_prefix_decode.py"),
)
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(H.manager_mod, "KVTransferManager", H._StubTransfer)


@pytest.fixture
def eager(monkeypatch):
    """Extend the chains on every step, as before the lazy batching."""
    monkeypatch.setenv("MSTAR_KV_CHAIN_LAZY_STEPS", "1")


def _chain_state(kv, rid):
    stream = kv._streams[rid]["main"]
    chain = stream.chain
    if chain is None:
        return None
    return (chain.covered_len, list(chain.keys) if hasattr(chain, "keys") else None,
            chain.unkeyed if hasattr(chain, "unkeyed") else None)


def _run(batched: bool, lazy_steps: int | None = None, flush: bool = False):
    kv = H._manager()
    if lazy_steps is not None:
        kv._chain_lazy_steps = lazy_steps
    rids = ["r0", "r1"]
    prompts = {"r0": list(range(100)), "r1": list(range(50, 50 + H.PAGE_SIZE + 10))}
    for rid in rids:
        H._ingest(kv, rid, prompts[rid])
    # the prefill of each, and the token it sampled
    for rid in rids:
        H._step(kv, rid, len(prompts[rid]))
    sampled = {"r0": 9000, "r1": 9100}
    if batched:
        rows = HostRows(request_ids=tuple(rids), buffers={
            H.TENSOR: torch.tensor([[sampled[r]] for r in rids]),
        })
        kv.extend_prefix_chains_batch(rids, H.NODE, H.WALK, rows)
    else:
        for rid in rids:
            kv.extend_prefix_chain(rid, H.NODE, H.WALK, {H.TENSOR: [torch.tensor([sampled[rid]])]})
    # decode steps: each writes the id sampled before it and samples a new one
    for t in range(1, 2 * H.PAGE_SIZE + 3):
        for rid in rids:
            H._step(kv, rid, 1)
        tokens = {"r0": 9000 + t, "r1": 9100 + t}
        if batched:
            rows = HostRows(request_ids=tuple(reversed(rids)), buffers={
                H.TENSOR: torch.tensor([[tokens[r]] for r in reversed(rids)]),
            })
            kv.extend_prefix_chains_batch(rids, H.NODE, H.WALK, rows)
        else:
            for rid in rids:
                kv.extend_prefix_chain(rid, H.NODE, H.WALK, {H.TENSOR: [torch.tensor([tokens[rid]])]})
    kv.assert_pages_conserved()
    if flush:
        kv.flush_chain_backlog()
    return H._indexed(kv), {rid: _chain_state(kv, rid) for rid in rids}


def test_batched_extension_keys_the_same_pages(eager):
    per_rid = _run(batched=False)
    batched = _run(batched=True)
    assert per_rid == batched
    assert per_rid[0] >= 3, "the generated pages were not indexed"


def test_lazy_extension_keys_the_same_pages_once_flushed(eager):
    per_rid = _run(batched=False)
    for lazy in (3, 7, H.PAGE_SIZE):
        lazy_run = _run(batched=True, lazy_steps=lazy)
        # the tokens of the last steps are still waiting, so flush them and
        # compare the chains; the pages a later commit would have indexed
        # from the eager form are a subset of what was indexed
        assert lazy_run[0] <= per_rid[0]
        kv_state = lazy_run[1]
        # every chain is at most `lazy` tokens behind and never ahead
        for rid in per_rid[1]:
            eager_len = per_rid[1][rid][0]
            lazy_len = kv_state[rid][0]
            assert eager_len - lazy < lazy_len <= eager_len, (rid, lazy, eager_len, lazy_len)
        # once the waiting steps go out the chains are the eager ones
        flushed = _run(batched=True, lazy_steps=lazy, flush=True)
        assert flushed[1] == per_rid[1]
        assert flushed[0] <= per_rid[0]


def test_lazy_extension_flushes_at_the_threshold_and_on_removal(eager):
    kv = H._manager()
    kv._chain_lazy_steps = 4
    rids = ["r0", "r1"]
    prompts = {"r0": list(range(H.PAGE_SIZE)), "r1": list(range(50, 50 + H.PAGE_SIZE))}
    for rid in rids:
        H._ingest(kv, rid, prompts[rid])
        H._step(kv, rid, len(prompts[rid]))
    before = {rid: _chain_state(kv, rid) for rid in rids}

    def step(t, order):
        rows = HostRows(request_ids=tuple(order), buffers={
            H.TENSOR: torch.tensor([[9000 + t if r == "r0" else 9100 + t] for r in order]),
        })
        kv.extend_prefix_chains_batch(rids, H.NODE, H.WALK, rows)

    # three steps wait in the backlog, the chains do not move
    for t in range(3):
        step(t, rids)
    assert {rid: _chain_state(kv, rid) for rid in rids} == before
    assert len(kv._chain_backlog) == 3
    # the fourth step flushes: both runs (the rows change order at t=2) land
    step(3, list(reversed(rids)))
    assert kv._chain_backlog == []
    assert _chain_state(kv, "r0")[2] == [9000, 9001, 9002, 9003]
    assert _chain_state(kv, "r1")[2] == [9100, 9101, 9102, 9103]
    # a removal flushes what waits before the request goes
    step(4, rids)
    step(5, rids)
    kv.remove_request("r1")
    assert kv._chain_backlog == []
    assert _chain_state(kv, "r0")[2] == [9000, 9001, 9002, 9003, 9004, 9005]
    assert "r1" not in kv._streams
    # a flush with nothing waiting is a no-op
    kv.flush_chain_backlog()
    assert _chain_state(kv, "r0")[2] == [9000, 9001, 9002, 9003, 9004, 9005]


def test_batched_extension_skips_rows_it_does_not_have(eager):
    kv = H._manager()
    prompt = list(range(H.PAGE_SIZE))
    H._ingest(kv, "r0", prompt)
    H._step(kv, "r0", len(prompt))
    before = _chain_state(kv, "r0")
    # no buffer under the keyed tensor name, and a request with no row
    kv.extend_prefix_chains_batch(["r0", "ghost"], H.NODE, H.WALK,
                                  HostRows(request_ids=("r0",), buffers={"other": torch.tensor([[1]])}))
    kv.extend_prefix_chains_batch(["r0"], H.NODE, H.WALK,
                                  HostRows(request_ids=("r1",), buffers={H.TENSOR: torch.tensor([[1]])}))
    assert _chain_state(kv, "r0") == before


def test_runner_dispatch_falls_back_per_request():
    calls = []

    class _Batch:
        def extend_prefix_chains_batch(self, rids, node, walk, host_rows):
            calls.append(("batch", tuple(rids)))

    class _PerRid:
        def extend_prefix_chain(self, rid, node, walk, outputs):
            calls.append(("rid", rid, outputs))

    runner = StepRunner.__new__(StepRunner)
    runner._resources = {"a": _Batch(), "b": _PerRid()}
    outputs = {"r0": {"x": 1}, "r1": None}
    runner.extend_prefix_chains_batch(["r0", "r1"], "n", "w", HostRows((), {}), outputs, keys=["a", "b"])
    assert calls == [("batch", ("r0", "r1")), ("rid", "r0", {"x": 1})]
