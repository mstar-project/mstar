"""``SamplerBuffers.gather_static`` skips a steady batch even while requests
outside the batch wait for their slot init (registered, still queued)."""
import pytest
import torch

from mstar.engine.resources.sampler.utils import SamplerBuffers, SamplingConfig


def _buffers(monkeypatch) -> tuple[SamplerBuffers, list]:
    buffers = SamplerBuffers.allocate(8, torch.device("cpu"), cg_slots=2)
    gathers: list = []
    for buf in buffers._scalar_buffers():
        real = buf.gather

        def counting(rows, padded_bs, cg_slot, _real=real):
            gathers.append(cg_slot)
            return _real(rows, padded_bs, cg_slot)

        monkeypatch.setattr(buf, "gather", counting)
    return buffers, gathers


def test_queued_request_does_not_block_the_skip(monkeypatch):
    buffers, gathers = _buffers(monkeypatch)
    buffers.register_request("a", SamplingConfig(temperature=0.5))
    buffers.register_request("b", SamplingConfig(temperature=0.7))
    buffers.gather_static(["a", "b"], 4, 0)
    first = len(gathers)
    assert first > 0

    # a request registered but not yet in the batch: its slot awaits init
    buffers.register_request("c", SamplingConfig(temperature=0.9))
    assert buffers._pending_init
    buffers.gather_static(["a", "b"], 4, 0)
    assert len(gathers) == first, "steady batch gathered again"
    assert buffers._last_real_bs[0] == 2

    # once it joins the batch, the gather runs and inits its row
    buffers.gather_static(["a", "b", "c"], 4, 0)
    assert len(gathers) > first
    assert not buffers._pending_init
    slot_c = buffers._rid_to_slot["c"]
    assert float(buffers.temperature.master[slot_c]) == pytest.approx(0.9)


def test_pending_row_in_the_batch_forces_the_gather(monkeypatch):
    buffers, gathers = _buffers(monkeypatch)
    buffers.register_request("a", SamplingConfig(temperature=0.5))
    buffers.register_request("b", SamplingConfig(temperature=0.7))
    buffers.gather_static(["a", "b"], 4, 0)
    first = len(gathers)

    # re-registered onto a new slot: the key still matches, the row does not
    buffers.unregister_request("a")
    buffers.register_request("a", SamplingConfig(temperature=0.2))
    buffers.gather_static(["a", "b"], 4, 0)
    assert len(gathers) > first
    slot_a = buffers._rid_to_slot["a"]
    assert float(buffers.temperature.master[slot_a]) == pytest.approx(0.2)


def test_config_change_and_other_slot_still_gather(monkeypatch):
    buffers, gathers = _buffers(monkeypatch)
    buffers.register_request("a", SamplingConfig(temperature=0.5))
    buffers.gather_static(["a"], 2, 0)
    first = len(gathers)
    buffers.gather_static(["a"], 2, 1)  # the other cg slot has nothing staged
    second = len(gathers)
    assert second > first
    buffers.gather_static(["a"], 2, 1)
    assert len(gathers) == second
    buffers.update_request_config("a", SamplingConfig(temperature=0.1))
    buffers.gather_static(["a"], 2, 1)
    assert len(gathers) > second
