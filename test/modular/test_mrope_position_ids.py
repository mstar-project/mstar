"""Where each row of a packed MRoPE step reads its positions.

Qwen3.5 and Qwen3-Omni's Thinker pack text, image and audio rows into one step.
A text row's positions come from the position resource's plan, which covers every
row of the step in packed order, so a text row after an image must read the plan
past the image's tokens, not where the previous text row ended.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import torch

from mstar.model.components.mrope import mrope_position_ids
from mstar.model.submodule_base import ARNodeInputs


def test_a_text_row_after_an_image_reads_the_plan_past_the_images_tokens():
    plan = torch.arange(10, 19)
    image = torch.tensor([[5.0, 5.0], [5.0, 6.0], [5.0, 7.0]])
    rows = [
        ARNodeInputs(input_seq_len=3),
        ARNodeInputs(input_seq_len=2, custom_pos_ids=image),
        ARNodeInputs(input_seq_len=4),
    ]

    positions = mrope_position_ids(plan, rows)

    assert positions.tolist() == [
        [10, 11, 12, *grid, 15, 16, 17, 18] for grid in image.tolist()
    ], "a row took positions from the wrong slice of the plan, so its tokens rotate as if elsewhere in the prompt"
