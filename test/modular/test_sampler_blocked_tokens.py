"""Per-step blocking on eager and graph sampler dispatch, emulated on CPU."""

import pytest
import torch

from mstar.engine.resources.base import EngineResourceInfo, build_resource
from mstar.engine.resources.sampler.config import SamplerSpec
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.engine.resources.sampler.utils import AcceleratorGraphableSampler

CPU = torch.device("cpu")


def _resource(enabled: bool) -> SamplerResource:
    spec = SamplerSpec(
        resource_key="sampler", nodes={"LM"}, vocab_size=None,
        enable_repetition_penalty=False, enable_blocked_tokens=enabled,
    )
    resource = build_resource(spec, EngineResourceInfo(device=CPU))
    for rid in ("a", "b"):
        resource._sampler.add_request(rid)
        resource._sampler.set_config(rid, temperature=0)
    resource._greedy_this_step = True
    return resource


@pytest.mark.parametrize("graph_sampler_path", [False, True])
def test_blocked_rows_change_the_draw_without_changing_logits(graph_sampler_path):
    resource = _resource(True)
    if graph_sampler_path:
        resource._cg_sampler = AcceleratorGraphableSampler(
            temperature_buf=torch.zeros(2), top_k_buf=torch.zeros(2, dtype=torch.int32),
            top_p_buf=torch.ones(2), seed_buf=torch.zeros(2, dtype=torch.long),
            offset_buf=torch.zeros(2, dtype=torch.long),
        )
    logits = torch.tensor([[9.0, 8.0, 1.0], [9.0, 8.0, 1.0]])
    blocked = torch.tensor([[True, False, False], [True, True, False]])
    assert resource.sample(["a", "b"], logits, blocked_tokens=blocked).tolist() == [1, 2]
    assert logits[:, 0].tolist() == [9.0, 9.0]
    assert resource.sample(["a", "b"], logits).tolist() == [0, 0]


def test_blocked_tokens_require_a_declared_capability_and_valid_layout():
    logits = torch.tensor([[3.0, 2.0], [3.0, 2.0]])
    blocked = torch.tensor([[True, False], [False, True]])
    with pytest.raises(ValueError, match="does not declare"):
        _resource(False).sample(["a", "b"], logits, blocked_tokens=blocked)
    resource = _resource(True)
    with pytest.raises(ValueError, match="shape and device"):
        resource.sample(["a", "b"], logits, blocked_tokens=blocked[:1])
    with pytest.raises(ValueError, match="bool tensor"):
        resource.sample(["a", "b"], logits, blocked_tokens=blocked.float())
