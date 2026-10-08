"""A Qwen3-Omni vision prefill runs in chunks like text and audio: each chunk
cuts its deepstack and talker masks with its tokens, and the MRoPE position
counter, which a vision span advances by its 3D-grid extent rather than its
token count, moves once, on the final chunk."""
from dataclasses import replace

import pytest
import torch

from mstar.model.qwen3_omni.submodules import ThinkerSubmodule
from mstar.model.submodule_base import ARNodeInputs

HIDDEN, N, ADVANCE = 4, 10, 37


@pytest.fixture
def thinker():
    return object.__new__(ThinkerSubmodule)


def _vision():
    return ARNodeInputs(
        input_seq_len=N,
        input_embeds=torch.arange(N * HIDDEN, dtype=torch.float).reshape(N, HIDDEN),
        custom_pos_ids=torch.arange(3 * N, dtype=torch.float).reshape(3, N),
        tensor_inputs={
            "masks_for_talker": torch.arange(2 * N).reshape(2, N),
            "deepstack": [torch.full((N, HIDDEN), float(i)) + torch.arange(N)[:, None] for i in range(3)],
            "mrope_pos_advance": ADVANCE,
        },
    )


def test_a_vision_chunk_cuts_every_sequence_input(thinker):
    cut = thinker.split_inputs("prefill_vision", None, _vision(), 3, 7)

    assert cut.input_seq_len == 4
    assert cut.input_embeds.shape == (4, HIDDEN)
    assert cut.tensor_inputs["masks_for_talker"].tolist() == [[3, 4, 5, 6], [13, 14, 15, 16]]
    assert [layer[:, 0].tolist() for layer in cut.tensor_inputs["deepstack"]] == [
        [i + 3.0, i + 4.0, i + 5.0, i + 6.0] for i in range(3)
    ]
    assert cut.tensor_inputs["mrope_pos_advance"] == ADVANCE


def test_an_input_it_cannot_cut_is_refused(thinker):
    inputs = _vision()
    inputs.tensor_inputs["unknown"] = torch.zeros(N)

    with pytest.raises(NotImplementedError, match="unknown"):
        thinker.split_inputs("prefill_vision", None, inputs, 0, 4)


@pytest.mark.parametrize("chunk_start,advance", [(0, 0), (6, ADVANCE)])
def test_only_the_final_vision_chunk_advances_positions(thinker, chunk_start, advance):
    from mstar.model.qwen3_omni.config import THINKER_POS

    cut = thinker.split_inputs("prefill_vision", None, _vision(), chunk_start, chunk_start + 4)
    cut = replace(cut, chunk_start=chunk_start, chunk_total=N)

    step = thinker.declare_step("prefill_vision", [0], [cut])

    assert step.steps[THINKER_POS].advance == (advance,)
