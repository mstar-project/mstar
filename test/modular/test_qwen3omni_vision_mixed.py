"""Qwen3-Omni vision prefill rows ride in the Thinker's mixed steps: deepstack
is packed over every row (zeros beside non-vision rows), each row advances
MRoPE positions by its own rule, and such a step replays captures that take
deepstack."""
from types import SimpleNamespace

import pytest
import torch

from mstar.model.qwen3_omni.config import THINKER_MIXED, THINKER_POS
from mstar.model.qwen3_omni.submodules import ThinkerSubmodule
from mstar.model.submodule_base import ARNodeInputs

HIDDEN = 4


@pytest.fixture
def thinker():
    thinker = object.__new__(ThinkerSubmodule)
    thinker.__dict__["config"] = SimpleNamespace(
        vision=SimpleNamespace(deepstack_visual_indexes=[1, 2]), thinker_hidden_size=HIDDEN,
    )
    return thinker


def _row(n, walk, deepstack=None, advance=None):
    tensor_inputs = {}
    if deepstack is not None:
        tensor_inputs = {"deepstack": deepstack, "mrope_pos_advance": advance}
    return ARNodeInputs(input_seq_len=n, graph_walk=walk, tensor_inputs=tensor_inputs)


def test_deepstack_is_packed_over_every_row(thinker):
    vision = [torch.full((3, HIDDEN), 1.0), torch.full((3, HIDDEN), 2.0)]
    rows = [_row(1, "thinker_decode"), _row(3, "prefill_vision", vision, 9), _row(2, "prefill_text")]

    out = thinker._deepstack_inputs(rows, torch.float, torch.device("cpu"))

    assert sorted(out) == ["deepstack_0", "deepstack_1"]
    assert out["deepstack_0"][:, 0].tolist() == [0, 1, 1, 1, 0, 0]
    assert out["deepstack_1"][:, 0].tolist() == [0, 2, 2, 2, 0, 0]


def test_each_row_of_a_mixed_step_advances_positions_by_its_own_rule(thinker):
    vision = [torch.zeros(3, HIDDEN)] * 2
    rows = [_row(1, "thinker_decode"), _row(3, "prefill_vision", vision, 9), _row(2, "prefill_text")]

    step = thinker.declare_step(THINKER_MIXED, [0, 1, 2], rows)

    assert step.steps[THINKER_POS].advance == (1, 9, 2)


@pytest.mark.parametrize("step_walk,walks,key", [
    (THINKER_MIXED, ["thinker_decode", "prefill_vision"], "deepstack"),
    (THINKER_MIXED, ["thinker_decode", "prefill_text"], None),
    ("prefill_vision", ["prefill_vision", "prefill_vision"], "deepstack"),
])
def test_a_step_with_a_vision_row_replays_the_deepstack_captures(thinker, step_walk, walks, key):
    info = {i: SimpleNamespace(graph_walk=w, step_metadata={}) for i, w in enumerate(walks)}

    assert thinker.cg_key_info(step_walk, info) == key


def test_one_vision_capture_serves_vision_and_mixed_steps(thinker):
    thinker._get_decode_thinker_mask = lambda device: torch.zeros(2, 1)
    configs = thinker.get_cuda_graph_configs(torch.device("cpu"))
    vision = [c for c in configs if "prefill_vision" in c.replay_graph_walks]

    assert len(vision) == 1
    assert vision[0].replay_graph_walks == ["prefill_vision", THINKER_MIXED]
    assert vision[0].additional_key_info == "deepstack"


@pytest.mark.parametrize("step_walk,walks", [
    (THINKER_MIXED, ["thinker_decode", "prefill_vision"]),
    (THINKER_MIXED, ["thinker_decode", "prefill_text"]),
    ("prefill_vision", ["prefill_vision", "prefill_vision"]),
])
def test_declare_step_stamps_the_key_cg_key_info_leased(thinker, step_walk, walks):
    vision = [torch.zeros(1, HIDDEN)] * 2
    rows = [_row(1, w, vision if w == "prefill_vision" else None, 4) for w in walks]
    info = {i: SimpleNamespace(graph_walk=w, step_metadata={}) for i, w in enumerate(walks)}

    step = thinker.declare_step(step_walk, list(range(len(rows))), rows)

    assert step.cg_key_info == thinker.cg_key_info(step_walk, info)
