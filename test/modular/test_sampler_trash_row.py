"""Master row 0 is the sampler's trash row, in both scatter modes.

No request is ever given it, so its config stays ``SamplingConfig()``. Padding
rows and unregistered rids read it, and every write on their behalf lands there.
CUDA graph capture rows alias it rather than taking a slot each: they used to
hold ``cg_slots x max_bs`` master rows per runner for the process lifetime,
doubling the masters, or using up the fixed in-graph capacity.

CPU-only: the buffers' gather/scatter run on CPU views.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.accelerator_graph_runner import DummyRowPool
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.engine.resources.sampler.utils import SamplerBuffers, SamplingConfig

CPU = torch.device("cpu")


def _bufs(ingraph: bool, max_bs: int = 4, slots: int = 8) -> SamplerBuffers:
    return SamplerBuffers.allocate(
        max_batch_size=max_bs, device=CPU, cg_slots=2,
        ingraph_scatter=ingraph, slots=slots,
    )


@pytest.mark.parametrize("ingraph", [False, True])
def test_no_request_is_given_row_zero(ingraph):
    bufs = _bufs(ingraph)
    cap = bufs._master_capacity
    assert bufs.offset.master.shape[0] == cap + 1
    for i in range(cap):
        bufs.register_request(f"r{i}")
    assert sorted(bufs._rid_to_slot.values()) == list(range(1, cap + 1))
    if ingraph:
        return
    # growing keeps row 0 out of the free list and hands out new rows only
    bufs.register_request("extra")
    assert bufs._master_capacity == 2 * cap
    assert bufs.temperature.master.shape[0] == 2 * cap + 1
    # the device masters follow on the gpu thread's next gather
    assert bufs.offset.master.shape[0] == cap + 1
    bufs._sync_device_capacity()
    assert bufs.offset.master.shape[0] == 2 * cap + 1
    assert bufs.last_token.master.shape[0] == 2 * cap + 1
    assert bufs._rid_to_slot["extra"] > cap
    assert 0 not in bufs._free_slots


def test_unregistered_rid_writes_the_trash_row_not_a_request():
    """Off the knob, the commit writeback covers every real row; an unregistered
    rid there used to clobber whichever request held slot 0."""
    bufs = _bufs(ingraph=False)
    for i in range(bufs._master_capacity):
        bufs.register_request(f"r{i}")
    bufs.gather_static(["r0", "ghost"], 4, 0)
    bufs.gather_dynamic(["r0", "ghost"], 4, 0)
    assert bufs._slot_idx_gpu[0, :4].tolist()[1:] == [0, 0, 0]
    bufs.offset.slot_view(0, 4).fill_(7)
    bufs.scatter_offset(0)
    owned = [bufs._rid_to_slot[f"r{i}"] for i in range(1, bufs._master_capacity)]
    assert bufs.offset.master[owned].tolist() == [0] * len(owned)
    assert bufs.offset.master[bufs._rid_to_slot["r0"]].item() == 7


def test_trash_row_keeps_the_default_config():
    bufs = _bufs(ingraph=False)
    bufs.register_trash_row("cap")
    bufs.update_request_config("cap", SamplingConfig(temperature=0.1, top_k=5))
    bufs.register_request("a", SamplingConfig(temperature=0.5))
    bufs.gather_static(["a", "cap"], 4, 0)
    temps = bufs.temperature.slot_view(0, 4).tolist()
    assert temps[0] == pytest.approx(0.5)
    assert temps[1:] == [1.0, 1.0, 1.0]


@pytest.mark.parametrize("ingraph", [False, True])
def test_capture_rows_take_no_master_rows(ingraph, monkeypatch):
    """The capture flow, verbatim: `ensure` per (config, cg slot) key, the
    `finally` reset, then `release_all`."""
    monkeypatch.setenv("MSTAR_SAMPLER_INGRAPH_SCATTER", "1" if ingraph else "0")
    monkeypatch.setenv("MSTAR_SAMPLER_SLOTS", "8")
    max_bs, vocab = 8, 32
    res = SamplerResource(vocab_size=vocab, enable_repetition_penalty=True, device=CPU)
    res._keep_last_token = ingraph
    res.build_accelerator_graph_buffers([SimpleNamespace(slot=s) for s in (0, 1)], max_bs, 1)
    bufs = res._cg_buffers
    assert bufs.ingraph_scatter == ingraph
    cap = bufs._master_capacity

    pool = DummyRowPool(
        "node", SimpleNamespace(ingest_capture_row=res.ingest_capture_row),
        {"sampler": res},
    )
    for slot in (0, 1):
        rids = pool.ensure(f"0_slot{slot}", max_bs)
        pool.reset(rids, free=True)
    pool.release_all()

    assert set(bufs._rid_to_slot.values()) == {0}
    assert len(bufs._free_slots) == cap
    # eager entries exist, so capture still tolerates the eager sampler
    assert all(
        m._seen_token_mask is None for m in res._sampler._seen_token_mask.values()
    )
    # a full machine of real requests fits without growing or raising
    for i in range(cap):
        res.ingest_request(f"real{i}")
    assert bufs._master_capacity == cap
    # removing a capture row frees nothing it never held
    res.remove_request(rids[0])
    assert len(bufs._free_slots) == 0
