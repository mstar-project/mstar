"""The planned expert flex BlockMask, including rectangular and captured rows."""

from __future__ import annotations

import pytest
import torch

from mstar.engine.resources.attn.ragged.config import (
    ExpertAttentionStep,
    RaggedAttentionConfig,
    RaggedExpertAttentionSpec,
)
from mstar.engine.resources.attn.ragged.expert_flex import FlexExpertWrapper
from mstar.engine.resources.attn.ragged.expert_mask import (
    ExpertCrossLayout,
    ExpertCrossMask,
    ExpertSelfLayout,
    ExpertSelfMask,
    cross_visibility,
    self_visibility,
)
from mstar.engine.resources.base import EngineResourceInfo, build_resource
from mstar.engine.resources.step import Segment, StepContext


def _dense(mask, q_len: int, kv_len: int) -> torch.Tensor:
    qb, kb = mask.BLOCK_SIZE
    out = torch.zeros((q_len, kv_len), dtype=torch.bool)
    for counts, indices, full in (
        (mask.kv_num_blocks, mask.kv_indices, False),
        (mask.full_kv_num_blocks, mask.full_kv_indices, True),
    ):
        if counts is None:
            continue
        for qblock in range(counts.shape[-1]):
            qs = torch.arange(qblock * qb, min((qblock + 1) * qb, q_len))
            for kblock in indices[0, 0, qblock, : int(counts[0, 0, qblock])].tolist():
                ks = torch.arange(kblock * kb, min((kblock + 1) * kb, kv_len))
                out[qs[:, None], ks[None, :]] = (
                    True if full else mask.mask_mod(0, 0, qs[:, None], ks[None, :])
                )
    return out


@pytest.mark.parametrize("mode", ["cross", "self"])
@pytest.mark.parametrize("layouts", [
    ((1, 2, 0, 2),), ((1, 2, 2, 2),), ((2, 2, 2, 4),),
    ((1, 2, 0, 2), (2, 2, 2, 4)),
])
def test_planned_block_mask_matches_oracle(mode: str, layouts) -> None:
    wrapper = FlexExpertWrapper(
        2, 2, 32, mode, 2, 1, 1, 2, True, 16, True,
        device=torch.device("cpu"), q_data_type=torch.float32,
    )
    wrapper.plan(layouts)
    q_spans = [k * 4 for k, _, _, _ in layouts]
    kv_spans = [q if mode == "self" else g + clean + n
                for q, (_, g, clean, n) in zip(q_spans, layouts, strict=True)]
    actual = _dense(wrapper.block_mask, sum(q_spans), sum(kv_spans))
    expected = torch.block_diag(*(
        self_visibility(ExpertSelfMask(2, 1), ExpertSelfLayout(k))
        if mode == "self" else cross_visibility(
            ExpertCrossMask(2, 1, 1, 2), ExpertCrossLayout(k, g, clean, n),
        ) for k, g, clean, n in layouts
    ))
    assert torch.equal(actual, expected)


def test_graph_mode_padding_and_replan_do_not_expose_other_request() -> None:
    wrapper = FlexExpertWrapper(
        2, 2, 32, "cross", 2, 1, 1, 2, True, 16, False,
        max_num_segments=2, max_total_tokens=32, accelerator_graph=True,
        device=torch.device("cpu"), q_data_type=torch.float32,
    )
    first = ((2, 2, 2, 4),)
    second = ((1, 2, 0, 2), (2, 2, 2, 4))
    wrapper.plan(first)
    assert torch.equal(
        _dense(wrapper.block_mask, 32, 32)[:8, :8],
        cross_visibility(ExpertCrossMask(2, 1, 1, 2), ExpertCrossLayout(2, 2, 2, 4)),
    )
    wrapper.plan(second)
    dense = _dense(wrapper.block_mask, 32, 32)
    assert not dense[:4, 4:12].any()
    assert not dense[4:12, :4].any()
    assert not dense[:12, 12:].any()
    q, k, v = (torch.randn(32, 2, 32) for _ in range(3))
    out = wrapper.run(q, k, v)
    assert not out[12:].any()
    wrapper.forget_plan()
    wrapper.plan(first)
    assert wrapper.num_segments == 1


def test_capture_dummy_segment_is_isolated_from_real_queries() -> None:
    wrapper = FlexExpertWrapper(
        2, 2, 32, "cross", 2, 1, 1, 2, True, 16, False,
        device=torch.device("cpu"), q_data_type=torch.float32,
    )
    wrapper.plan(((1, 2, 0, 2), (0, 4, 5, 0)))
    dense = _dense(wrapper.block_mask, 8, 9)
    assert torch.equal(
        dense[:4, :4], cross_visibility(ExpertCrossMask(2, 1, 1, 2), ExpertCrossLayout(1, 2, 0, 2)),
    )
    assert not dense[:4, 4:].any()


def test_capture_token_ceiling_checks_both_rectangular_sides() -> None:
    wrapper = FlexExpertWrapper(
        2, 2, 32, "cross", 2, 1, 1, 2, True, 16, False,
        max_num_segments=1, max_total_tokens=7, accelerator_graph=True,
        device=torch.device("cpu"), q_data_type=torch.float32,
    )
    # Four queries fit, but eight keys exceed the captured bucket's width.
    with pytest.raises(ValueError, match="spans 4 Q and 8 KV, ceilings 7/7"):
        wrapper.plan(((1, 6, 0, 2),))


def test_manager_checks_paired_spans_before_planning() -> None:
    spec = RaggedExpertAttentionSpec(
        resource_key="expert", nodes={"expert"},
        config=RaggedAttentionConfig(num_qo_heads=2, num_kv_heads=2, head_dim=32),
        mode="cross", chunk_len=2, future_per_chunk=1, tokens_per_frame=1, window_frames=2,
    )
    manager = build_resource(spec, EngineResourceInfo(device=torch.device("cpu"), kv_dtype=torch.float32))
    ctx = StepContext(request_ids=["r"], graph_walk="chunk", slot=0, capture=False)
    step = ExpertAttentionStep(
        segments=(Segment("r", "q", 8), Segment("r", "kv", 8)),
        pairs=(("q", "kv"),), layouts=((2, 2, 2, 4),),
    )
    manager.plan(step, ctx)
    assert manager.num_segments("q<-kv") == 1
    with pytest.raises(ValueError, match="spans Q"):
        manager.plan(ExpertAttentionStep(
            segments=(Segment("r", "q", 7), Segment("r", "kv", 8)),
            pairs=(("q", "kv"),), layouts=((2, 2, 2, 4),),
        ), ctx)
