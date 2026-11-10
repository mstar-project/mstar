"""GLM-5.2's long-context startup checks."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mstar.model.glm52.config import Glm52ModelConfig  # noqa: E402
from mstar.model.glm52.submodules import Glm52LLMSubmodule  # noqa: E402


def _submodule(config) -> Glm52LLMSubmodule:
    sub = object.__new__(Glm52LLMSubmodule)
    sub.config = config
    return sub


def test_long_context_bind_refuses_an_empty_prefill_window():
    cfg = Glm52ModelConfig(dsa_long_context=True, max_seq_len=262144, prefill_window_tokens=0)
    with pytest.raises(ValueError, match="must be >= 1"):
        _submodule(cfg).bind_node_resources({})


@pytest.mark.parametrize("serving_ladder, captured", [
    ({}, [32, 64]),  # the model's own ladder
    ({"prefill": [32, 128]}, [32]),  # the serving config's
])
def test_no_ladder_captures_a_prefill_bucket_past_the_window(serving_ladder, captured):
    """A prefill step past prefill_window_tokens plans no dense attention and runs eager in
    windows, so the engine must capture no bucket that large, wherever its ladder came from."""
    import types

    import torch

    from mstar.engine.accelerator_graph_runner import AcceleratorGraphRunner

    language_model = torch.nn.Module()
    language_model.lm_head = torch.nn.Identity()
    sub = Glm52LLMSubmodule(language_model, Glm52ModelConfig(
        dsa_long_context=True, mla_absorb=True, max_seq_len=262144, prefill_window_tokens=64))
    sub._moe_capture_blocked = lambda tp_world_size: False
    runner = AcceleratorGraphRunner(
        submodule_name="LLM", submodule=sub, resources={}, step_runner=None,
        device=torch.device("cpu"), autocast_dtype=None,
        joint_comm_group=types.SimpleNamespace(world_size=1), num_slots=1,
        token_ladders=serving_ladder, step_tokens=lambda walk, rows: None)
    got = sorted({spec.bucket.num_tokens for spec in runner._get_slot_specs()
                  if spec.bucket.graph_walk == "prefill" and spec.bucket.bs == 1})
    assert got == captured

