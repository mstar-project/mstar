"""The seen-token mask master is vocab-wide, so with fixed masters (a walk
with persistent rows) it still starts at a step's rows and grows only to the
slots in use, which are handed out lowest first."""
import pytest

torch = pytest.importorskip("torch")

from mstar.engine.resources.sampler.utils import SamplerBuffers  # noqa: E402


def test_the_mask_master_grows_to_the_slots_in_use():
    bufs = SamplerBuffers.allocate(4, torch.device("cpu"), vocab_size=16, fixed_masters=True, slots=64)
    assert bufs._master_capacity == 64
    assert len(bufs.seen_tokens.master) == 5
    for rid in range(10):
        bufs.register_request(f"r{rid}")
    assert sorted(bufs._rid_to_slot.values()) == list(range(1, 11))
    bufs.seen_tokens.master[3, 2] = True
    bufs._sync_device_capacity()
    assert 11 <= len(bufs.seen_tokens.master) < 65
    assert bufs.seen_tokens.master[3, 2]
    # a freed slot is the next one handed out
    bufs.unregister_request("r2")
    bufs.register_request("r10")
    assert bufs._rid_to_slot["r10"] == 3
