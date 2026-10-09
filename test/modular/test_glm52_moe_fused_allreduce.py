"""The GLM-5.2 MoE fused all-reduce comes from model_kwargs, so a serve yaml
can turn it on; a set MSTAR_GLM52_MOE_FUSED_ALLREDUCE still overrides it."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from test_glm52_moe import _RecordingGroup  # noqa: E402

from mstar.model.glm52.components.moe import Glm52SparseMoeBlock  # noqa: E402
from mstar.model.glm52.glm52_model import Glm52Model  # noqa: E402

ENV = "MSTAR_GLM52_MOE_FUSED_ALLREDUCE"


@pytest.mark.parametrize("config_on, env, fused", [
    (False, None, False), (True, None, True),
    (True, "0", False), (False, "1", True),
])
def test_fused_allreduce_from_model_kwargs(monkeypatch, config_on, env, fused):
    if env is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, env)
    model = Glm52Model(
        model_path_hf="", tokenizer_mode="byte", config_variant="reduced",
        moe_fused_allreduce=config_on,
    )
    assert model.config.moe_fused_allreduce is config_on
    block = Glm52SparseMoeBlock(model.config, comm_group=_RecordingGroup())
    assert block._fused_allreduce is fused
    assert block.shared_expert.down_proj.reduce_results is not fused


def test_single_rank_never_fuses(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    model = Glm52Model(
        model_path_hf="", tokenizer_mode="byte", config_variant="reduced",
        moe_fused_allreduce=True,
    )
    block = Glm52SparseMoeBlock(model.config, comm_group=_RecordingGroup(world_size=1))
    assert not block._fused_allreduce


def test_off_by_default(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    model = Glm52Model(model_path_hf="", tokenizer_mode="byte", config_variant="reduced")
    assert not model.config.moe_fused_allreduce
    assert not Glm52SparseMoeBlock(model.config, comm_group=_RecordingGroup())._fused_allreduce
