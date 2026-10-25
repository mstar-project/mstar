"""Qwen3-Omni Thinker postprocess: what each row streams to the Talker.

A decode row skips the boolean selection the prefill rows go through (it would
stall the GPU thread on every step) and must stream exactly what that selection
gives for its one text token, in memory of its own.
"""
from types import SimpleNamespace

import torch

from mstar.model.qwen3_omni.submodules import ThinkerSubmodule

HIDDEN = 4


def _thinker() -> ThinkerSubmodule:
    thinker = object.__new__(ThinkerSubmodule)
    thinker.__dict__["config"] = SimpleNamespace(thinker_hidden_size=HIDDEN)
    return thinker


def _info(walk: str) -> SimpleNamespace:
    return SimpleNamespace(graph_walk=walk, step_metadata={"audio_output": True, "is_last_prefill": False})


def _outputs(states: torch.Tensor, mask: torch.Tensor) -> dict:
    return {"thinker_states": [states], "thinker_mask": [mask], "new_token": [torch.tensor([7])]}


def test_a_decode_row_streams_what_the_selection_gives():
    thinker = _thinker()
    packed = torch.arange(3 * 2 * HIDDEN, dtype=torch.float).view(3, 2 * HIDDEN)
    decode_mask = torch.tensor([[0], [1]], dtype=torch.bool)   # text, kept
    row = packed[1:2]   # a decode row's view into a batch's output

    fast = _outputs(row, decode_mask)
    thinker.postprocess("r", _info("thinker_decode"), fast)
    general = _outputs(row, decode_mask)
    thinker.postprocess("r", _info("prefill_text"), general)

    assert torch.equal(fast["thinker_states"][0], general["thinker_states"][0])
    assert torch.equal(fast["thinker_states"][0], packed[1:2, :HIDDEN])
    assert fast["thinker_mask"][0].dtype == general["thinker_mask"][0].dtype == torch.bool
    assert torch.equal(fast["thinker_mask"][0], general["thinker_mask"][0])
    assert fast["text_inputs"] is fast["new_token"]
    # owned copies: a captured graph's next replay must not change what was streamed
    assert fast["thinker_states"][0].data_ptr() != row.data_ptr()
    assert fast["thinker_mask"][0].data_ptr() != decode_mask.data_ptr()


def test_a_prefill_row_still_drops_the_positions_the_talker_skips():
    thinker = _thinker()
    states = torch.arange(4 * 2 * HIDDEN, dtype=torch.float).view(4, 2 * HIDDEN)
    # token 0 excluded (system prompt), token 2 multimodal, tokens 1 and 3 text
    mask = torch.tensor([[0, 0, 1, 0], [0, 1, 0, 1]], dtype=torch.bool)
    outputs = _outputs(states, mask)

    thinker.postprocess("r", _info("prefill_vision"), outputs)

    kept = outputs["thinker_states"][0]
    assert torch.equal(kept[0], states[1, :HIDDEN])
    assert torch.equal(kept[1], states[2, HIDDEN:])
    assert torch.equal(kept[2], states[3, :HIDDEN])
    assert outputs["thinker_mask"][0].tolist() == [False, True, False]
