"""CPU checks of the proposed expert masks against the pinned Praxis reference."""

from __future__ import annotations

import pytest
import torch

from mstar.engine.resources.attn.ragged.expert_mask import (
    ExpertCrossLayout,
    ExpertCrossMask,
    ExpertSelfLayout,
    ExpertSelfMask,
    cross_visibility,
    self_visibility,
)


@pytest.mark.parametrize("include_noised", [False, True])
@pytest.mark.parametrize("num_chunks,clean_frames,window_frames", [
    (1, 0, 2),   # cold serving
    (1, 2, 2),   # warm serving: one noised chunk follows real clean frames
    (2, 2, 2),   # training: noised and clean chunks overlap in time
])
def test_cross_and_self_match_reference(
    include_noised: bool, num_chunks: int, clean_frames: int, window_frames: int,
) -> None:
    ref = pytest.importorskip("praxis.action_expert")
    c, f, g, t = 2, 1, 2, 1
    n = num_chunks * c
    cross, self_, q_len, kv_len = ref.trunk_mask_mods(
        num_action_tokens=n, num_future_per_chunk=f, chunk_size=c,
        num_clean_frames=clean_frames, num_noised_frames=n, tokens_per_frame=t,
        window_size=window_frames, num_global_tokens=g, device=torch.device("cpu"),
        include_noised=include_noised,
    )
    qi = torch.arange(q_len)[:, None]
    ki = torch.arange(kv_len)[None, :]
    expected_cross = cross(None, None, qi, ki)
    expected_self = self_(None, None, qi, torch.arange(q_len)[None, :])
    mask = ExpertCrossMask(c, f, t, window_frames, include_noised)
    layout = ExpertCrossLayout(num_chunks, g, clean_frames, n)
    actual_cross = cross_visibility(mask, layout)
    actual_self = self_visibility(ExpertSelfMask(c, f), ExpertSelfLayout(num_chunks))
    assert actual_cross.shape == (q_len, kv_len)
    assert torch.equal(actual_cross, expected_cross)
    assert torch.equal(actual_self, expected_self)


def test_two_chunk_hand_matrix_and_cold_case() -> None:
    cross = cross_visibility(
        ExpertCrossMask(2, 1, 1, 2), ExpertCrossLayout(2, 2, 2, 4)
    )
    same = self_visibility(ExpertSelfMask(2, 1), ExpertSelfLayout(2))
    assert cross.shape == (8, 8)
    for qi in (0, 2, 4, 5):
        assert cross[qi].nonzero().flatten().tolist() == [0, 1, 4, 5]
        assert same[qi].nonzero().flatten().tolist() == [0, 2, 4, 5]
    for qi in (1, 3, 6, 7):
        assert cross[qi].nonzero().flatten().tolist() == [0, 1, 2, 3, 6, 7]
        assert same[qi].nonzero().flatten().tolist() == [1, 3, 6, 7]

    cold = cross_visibility(ExpertCrossMask(2, 1, 1, 2), ExpertCrossLayout(1, 2, 0, 2))
    assert cold.shape == (4, 4)
    assert bool(cold.all())
    assert bool(self_visibility(ExpertSelfMask(2, 1), ExpertSelfLayout(1)).all())


def test_rejects_misaligned_or_unsupported_layout() -> None:
    mask = ExpertCrossMask(2, 1, 1, 2)
    with pytest.raises(ValueError, match="num_chunks"):
        cross_visibility(mask, ExpertCrossLayout(2, 2, 0, 2))
    with pytest.raises(ValueError, match="unsupported expert frame layout"):
        cross_visibility(mask, ExpertCrossLayout(2, 2, 1, 4))
    with pytest.raises(ValueError, match="include_noised"):
        ExpertCrossMask(2, 1, 1, 2, 1)
