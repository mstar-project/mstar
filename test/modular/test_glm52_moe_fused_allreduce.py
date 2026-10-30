"""The GLM-5.2 MoE block reduces its two partials once under kernels: auto, and each apart
under the reference."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from test_glm52_moe import _RecordingGroup  # noqa: E402

from mstar.model.glm52.components.moe import Glm52SparseMoeBlock  # noqa: E402
from mstar.model.glm52.glm52_model import Glm52Model  # noqa: E402


@pytest.mark.parametrize("kernels", ["auto", "reference"])
def test_fused_allreduce_follows_kernels(kernels):
    model = Glm52Model(
        model_path_hf="", tokenizer_mode="byte", config_variant="reduced", kernels=kernels,
    )
    fused = kernels == "auto"
    block = Glm52SparseMoeBlock(model.config, comm_group=_RecordingGroup())
    assert block._fused_allreduce is fused
    assert block.shared_expert.down_proj.reduce_results is not fused


def test_single_rank_never_fuses():
    model = Glm52Model(model_path_hf="", tokenizer_mode="byte", config_variant="reduced")
    block = Glm52SparseMoeBlock(model.config, comm_group=_RecordingGroup(world_size=1))
    assert not block._fused_allreduce
