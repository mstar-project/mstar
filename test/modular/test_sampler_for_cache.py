"""``SamplerBuffers.sampler_for`` hands back one sampler per (bucket, slot)
over the same buffer views."""
import torch

from mstar.engine.resources.sampler.utils import SamplerBuffers


def test_sampler_for_is_cached_per_bucket_and_slot():
    buffers = SamplerBuffers.allocate(8, torch.device("cpu"), cg_slots=2)
    a = buffers.sampler_for(4, 0)
    assert buffers.sampler_for(4, 0) is a
    b = buffers.sampler_for(4, 1)
    c = buffers.sampler_for(8, 0)
    assert b is not a and c is not a and c is not b
    assert a.temperature_buf.data_ptr() == buffers.temperature.slot_view(0, 4).data_ptr()
    assert b.temperature_buf.data_ptr() == buffers.temperature.slot_view(1, 4).data_ptr()
    assert a.temperature_buf.shape[0] == 4 and c.temperature_buf.shape[0] == 8
    a.applied_penalty_in_graph = True
    assert buffers.sampler_for(4, 0).applied_penalty_in_graph is False
