"""A packed capture's rows follow its ``token_layout``: by default the tokens
spread evenly, and a mixed step's layout is one-token rows then a prompt."""

import torch

from mstar.engine.accelerator_graph_config import (
    PackedAcceleratorGraphConfig,
    decode_rows_then_prompt,
    distribute_tokens,
)
from mstar.model.submodule_base import ARNodeInputs


def test_decode_rows_then_one_prompt_of_the_rest():
    assert decode_rows_then_prompt(64, 4) == [1, 1, 1, 61]
    assert decode_rows_then_prompt(4, 4) == [1, 1, 1, 1]


def test_fewer_tokens_than_rows_spread_as_usual():
    assert decode_rows_then_prompt(3, 4) == distribute_tokens(3, 4)


def test_the_capture_rows_take_the_layout():
    cfg = PackedAcceleratorGraphConfig(
        capture_graph_walk="mixed",
        capture_token_lengths=[64],
        make_node_input=lambda n: ARNodeInputs(input_ids=torch.zeros(n, dtype=torch.long), input_seq_len=n),
        token_layout=decode_rows_then_prompt,
    )

    assert [row.input_seq_len for row in cfg.get_node_inputs(4, 64)] == [1, 1, 1, 61]
