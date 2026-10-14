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


def _chain_state(kv, rid):
    stream = kv._streams[rid]["main"]
    chain = stream.chain
    if chain is None:
        return None
    return (chain.covered_len, list(chain.keys) if hasattr(chain, "keys") else None,
            chain.unkeyed if hasattr(chain, "unkeyed") else None)


def _run(batched: bool):
    kv = H._manager()
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
    return H._indexed(kv), {rid: _chain_state(kv, rid) for rid in rids}


def test_batched_extension_keys_the_same_pages():
    per_rid = _run(batched=False)
    batched = _run(batched=True)
    assert per_rid == batched
    assert per_rid[0] >= 3, "the generated pages were not indexed"


def test_batched_extension_skips_rows_it_does_not_have():
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
