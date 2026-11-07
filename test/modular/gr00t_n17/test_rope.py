"""``mrope_position_ids``: where one request's tokens sit on M-RoPE's three grids.

Checked against an inline reference for transformers' ``Qwen3VLModel.get_rope_index``,
written from the algorithm rather than imported, so the test needs no checkpoint and no
transformers version pinned to ours.
"""

import pytest
import torch

from mstar.model.gr00t_n17.components.rope import mrope_position_ids

# Cosmos-Reason2's `<|image_pad|>`, and the merge every released GR00T size uses.
_IMAGE = 151655
_TEXT = 1
_MERGE = 2


def _prompt(spec: list) -> tuple[torch.Tensor, torch.Tensor]:
    """A spec of text-span lengths (``int``) and image grids (``(t, h, w)``) -> the two inputs."""
    tokens, grids = [], []
    for item in spec:
        if isinstance(item, int):
            tokens += [_TEXT] * item
        else:
            t, h, w = item
            tokens += [_IMAGE] * (t * (h // _MERGE) * (w // _MERGE))
            grids.append(item)
    return torch.tensor(tokens), torch.tensor(grids, dtype=torch.long).reshape(-1, 3)


# ----------------------------------------------------------------------------
# Inline reference: `Qwen3VLModel.get_rope_index` (`models/qwen3_vl/modeling_qwen3_vl.py`)
# for a single unpadded sequence. It walks the grids rather than the prompt: each one
# finds the next image token, the text before it gets `arange + st_idx`, and the block
# itself gets the (t, h', w') meshgrid offset past that text. `st_idx` is one past the
# largest index the previous block reached, which is what couples the three grids.
# ----------------------------------------------------------------------------


def _reference(input_ids: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
    tokens = input_ids.tolist()
    blocks: list[torch.Tensor] = []
    start = 0

    for grid in image_grid_thw.reshape(-1, 3).tolist():
        t, h, w = grid[0], grid[1] // _MERGE, grid[2] // _MERGE
        end = tokens.index(_IMAGE, start)
        text_len = end - start
        st_idx = int(blocks[-1].max()) + 1 if blocks else 0
        blocks.append(torch.arange(text_len).expand(3, -1) + st_idx)
        axes = torch.meshgrid(torch.arange(t), torch.arange(h), torch.arange(w), indexing="ij")
        blocks.append(torch.stack([axis.flatten() for axis in axes]) + text_len + st_idx)
        start = end + t * h * w

    if start < len(tokens):
        st_idx = int(blocks[-1].max()) + 1 if blocks else 0
        blocks.append(torch.arange(len(tokens) - start).expand(3, -1) + st_idx)
    return torch.cat(blocks, dim=1).float()


_CASES = {
    "text only": [7],
    "one image": [4, (1, 4, 6), 3],
    "an image at the very start": [(1, 4, 6), 5],
    "an image at the very end": [5, (1, 4, 6)],
    "nothing but an image": [(1, 4, 6)],
    "two images separated by text": [2, (1, 4, 6), 3, (1, 8, 4), 2],
    # h' != w' both ways round: the advance is max(h, w) // merge, not one axis
    "wider than tall": [3, (1, 4, 12), 2],
    "taller than wide": [3, (1, 12, 4), 2],
    # the shape the processor actually emits: 8x14 = 112 tokens a frame
    "the real frame": [6, (1, 16, 28), 4],
    "four real frames": [1, (1, 16, 28), 1, (1, 16, 28), 1, (1, 16, 28), 1, (1, 16, 28), 1],
}


@pytest.mark.parametrize("spec", _CASES.values(), ids=_CASES)
def test_the_positions_match_transformers_get_rope_index(spec):
    input_ids, grids = _prompt(spec)
    positions = mrope_position_ids(input_ids, grids, _IMAGE, _MERGE)
    assert positions.shape == (3, len(input_ids))
    assert torch.equal(positions, _reference(input_ids, grids))


# Parity cannot defend these: transformers 4.57 accepts the token-count mismatch, the
# run/grid count mismatch and t != 1, positioning each its own way. These cases are the
# only thing pinning the stricter contract, so each stands on its own.
_MALFORMED = {
    # the processor sends every frame as its own image, so a t > 1 grid is a wiring mistake
    "a multi-frame grid": ([_TEXT] + [_IMAGE] * 8, [[2, 4, 4]], r"t != 1"),
    "a run the grid cannot fill": ([_TEXT] + [_IMAGE] * 5, [[1, 4, 4]], r"5 image tokens for a grid worth 4"),
    "more runs than grids": (
        [_IMAGE] * 4 + [_TEXT] + [_IMAGE] * 4,
        [[1, 4, 4]],
        r"1 image grid\(s\) for more image span\(s\) than that",
    ),
    "more grids than runs": ([_TEXT] + [_IMAGE] * 4, [[1, 4, 4]] * 2, r"2 image grid\(s\) for 1 image span\(s\)"),
    # Two images with nothing between them read as one run, which no single grid accounts
    # for. Refusing is deliberate: transformers 4.57 gives the second image text positions and
    # 5.19 dies on an opaque shape error; a named ValueError beats both.
    "two runs glued together": ([_IMAGE] * 4 + [_IMAGE] * 6, [[1, 4, 4], [1, 4, 6]], r"10 image tokens"),
}


@pytest.mark.parametrize("tokens, grids, message", _MALFORMED.values(), ids=_MALFORMED)
def test_a_prompt_the_grids_do_not_describe_is_rejected(tokens, grids, message):
    with pytest.raises(ValueError, match=message):
        mrope_position_ids(torch.tensor(tokens), torch.tensor(grids), _IMAGE, _MERGE)
