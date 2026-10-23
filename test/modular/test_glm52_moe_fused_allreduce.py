"""The GLM-5.2 MoE fused all-reduce comes from model_kwargs, so a serve yaml
can turn it on."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from test_glm52_moe import _RecordingGroup  # noqa: E402

from mstar.model.glm52.components.moe import Glm52SparseMoeBlock  # noqa: E402
from mstar.model.glm52.glm52_model import Glm52Model  # noqa: E402


@pytest.mark.parametrize("config_on", [False, True])
def test_fused_allreduce_from_model_kwargs(config_on):
    model = Glm52Model(
        model_path_hf="", tokenizer_mode="byte", config_variant="reduced",
        moe_fused_allreduce=config_on,
    )
    assert model.config.moe_fused_allreduce is config_on
    block = Glm52SparseMoeBlock(model.config, comm_group=_RecordingGroup())
    assert block._fused_allreduce is config_on
    assert block.shared_expert.down_proj.reduce_results is not config_on


def test_single_rank_never_fuses():
    model = Glm52Model(
        model_path_hf="", tokenizer_mode="byte", config_variant="reduced",
        moe_fused_allreduce=True,
    )
    block = Glm52SparseMoeBlock(model.config, comm_group=_RecordingGroup(world_size=1))
    assert not block._fused_allreduce


def test_off_by_default():
    model = Glm52Model(model_path_hf="", tokenizer_mode="byte", config_variant="reduced")
    assert not model.config.moe_fused_allreduce
    assert not Glm52SparseMoeBlock(model.config, comm_group=_RecordingGroup())._fused_allreduce
