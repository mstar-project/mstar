"""A Qwen3-Omni vision or audio prefill runs in chunks like text: each chunk
cuts its deepstack and talker masks with its tokens, and the MRoPE position
counter, which a vision span advances by its 3D-grid extent rather than its
token count and an audio span past its whole clip, moves once, on the final
chunk."""
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


# ── audio ───────────────────────────────────────────────────────────────


def _audio():
    """An audio row, which carries its positions for the whole clip."""
    return ARNodeInputs(
        input_seq_len=N,
        input_embeds=torch.zeros(N, HIDDEN),
        custom_pos_ids=torch.arange(3 * N, dtype=torch.float).reshape(3, N),
        tensor_inputs={"masks_for_talker": torch.arange(2 * N).reshape(2, N)},
    )


@pytest.mark.parametrize("start,end,advance", [(0, 4, 0), (4, 8, 0), (8, N, N)])
def test_a_cut_audio_row_moves_past_its_whole_clip_once_on_its_final_chunk(thinker, start, end, advance):
    """Each chunk's positions are cut from the whole clip's, so moving the
    counter per chunk as well would count the clip twice."""
    from mstar.model.qwen3_omni.config import THINKER_POS

    cut = thinker.split_inputs("prefill_audio", None, _audio(), start, end)
    cut = replace(cut, chunk_start=start, chunk_total=N)

    step = thinker.declare_step("prefill_audio", [0], [cut])

    assert cut.custom_pos_ids[0].tolist() == list(map(float, range(start, end)))
    assert step.steps[THINKER_POS].advance == (advance,)


def test_an_uncut_audio_row_advances_by_its_span(thinker):
    from mstar.model.qwen3_omni.config import THINKER_POS

    step = thinker.declare_step("prefill_audio", [0], [_audio()])

    # None: the runner's default, the span it ran
    assert step.steps[THINKER_POS].advance is None


def test_a_non_final_chunk_still_streams_its_own_states_to_the_talker():
    """Every chunk streams its ``thinker_states`` and ``thinker_mask`` to the
    Talker (``per_chunk_outputs``), so postprocess has to produce them for a
    chunk that is not the prompt's last."""
    from types import SimpleNamespace

    thinker = object.__new__(ThinkerSubmodule)
    thinker.config = SimpleNamespace(thinker_hidden_size=HIDDEN)
    n = 4
    states = torch.cat([torch.zeros(n, HIDDEN), torch.ones(n, HIDDEN)], dim=-1)
    # tokens 0-1 multimodal (layer N), 2 kept text (layer 0), 3 dropped
    mask = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 0]])
    outputs = {"new_token": [torch.tensor([7])], "thinker_states": [states], "thinker_mask": [mask]}
    info = SimpleNamespace(graph_walk="prefill_audio", step_metadata={"is_last_prefill": False})

    thinker.postprocess(0, info, outputs)

    assert "new_token" not in outputs
    assert outputs["thinker_states"][0][:, 0].tolist() == [1.0, 1.0, 0.0]
    assert outputs["thinker_mask"][0].tolist() == [True, True, False]
