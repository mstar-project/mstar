"""Expert masks bind through declared state and match the CPU reference."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.accelerator_graph_runner import dummy_metadata
from mstar.engine.resources.attn.ragged.expert_mask import cross_visibility, self_visibility
from mstar.engine.resources.base import EngineResourceInfo, build_resource
from mstar.engine.resources.step import Segment, StepContext, SubmoduleStep
from mstar.engine.state import (
    Array,
    ArrayStep,
    Attend,
    ExpertCross,
    ExpertCrossLayout,
    ExpertSelf,
    ExpertSelfLayout,
    KVEntry,
    Read,
)
from mstar.engine.state.adapters import EXPERT_LAYOUT_KWARG, build_specs, node_handles, state_steps
from mstar.engine.state.analysis import _array_problems


@pytest.mark.parametrize("chunks,clean", [(1, 0), (1, 2), (2, 2)])
def test_declared_expert_reads_match_reference(chunks: int, clean: int) -> None:
    reference = pytest.importorskip("praxis.action_expert")
    c, f, g, t, window = 2, 1, 2, 1, 2
    noised = chunks * c
    q_len, kv_len = chunks * (1 + f + c), g + clean + noised
    q_mask, kv_mask = ExpertSelf(c, f), ExpertCross(c, f, t, window)
    arrays = [
        Array(
            "expert_q", nodes={"expert"}, labels={"q"}, layers={0},
            entry=KVEntry(heads=2, head_dim=32), max_len=0,
            reads={"self_attn": Read(mask=q_mask, op=Attend(q_heads=2))},
        ),
        Array(
            "trunk_kv", nodes={"expert"}, labels={"kv"}, layers={0},
            entry=KVEntry(heads=2, head_dim=32), max_len=0,
            reads={"cross_attn": Read(mask=kv_mask, op=Attend(q_heads=2), queries="expert_q")},
        ),
    ]
    assert all(_array_problems(array) == [] for array in arrays)
    specs = build_specs(arrays)
    assert len(specs) == 2
    resources = {
        spec.resource_key: build_resource(
            spec, EngineResourceInfo(device=torch.device("cpu"), kv_dtype=torch.float32),
        ) for spec in specs
    }
    model = SimpleNamespace(declare_state=lambda: arrays)
    handles = node_handles(model, "expert", resources)
    info = CurrentForwardPassInfo(
        request_id="r0", graph_walk="chunk", fwd_index=0, random_seed=0, max_tokens=1,
        step_metadata={EXPERT_LAYOUT_KWARG: {
            "expert_q": {"q": {"self_attn": [chunks]}},
            "trunk_kv": {"kv": {"cross_attn": [chunks, g, clean, noised]}},
        }},
    )
    step = SubmoduleStep(state={
        "expert_q": ArrayStep(segments=(Segment("r0", "q", q_len),)),
        "trunk_kv": ArrayStep(segments=(Segment("r0", "kv", kv_len),)),
    })
    planned = state_steps(step, handles, per_request_info={"r0": info})
    ctx = StepContext(request_ids=["r0"], graph_walk="chunk", slot=0, capture=False)
    for spec in specs:
        resources[spec.resource_key].plan(planned.steps[spec.resource_key], ctx)
    planned.set_ctx(ctx)
    for handle in handles.values():
        handle.begin_step(planned)
    torch.manual_seed(42 + chunks + clean)
    q, qk, qv = (torch.randn(q_len, 2, 32) for _ in range(3))
    k, v = (torch.randn(kv_len, 2, 32) for _ in range(2))
    handles["expert_q"].add(0, k=qk, v=qv)
    handles["trunk_kv"].add(0, k=k, v=v)
    actual_self = handles["expert_q"].read(0, q=q)
    actual_cross = handles["trunk_kv"].read(0, q=q)
    for handle in handles.values():
        handle.end_step(planned)

    cross_mod, self_mod, ref_q, ref_kv = reference.trunk_mask_mods(
        num_action_tokens=noised, num_future_per_chunk=f, chunk_size=c,
        num_clean_frames=clean, num_noised_frames=noised, tokens_per_frame=t,
        window_size=window, num_global_tokens=g, device=torch.device("cpu"),
    )
    assert (q_len, kv_len) == (ref_q, ref_kv)
    qi, ki = torch.arange(q_len)[:, None], torch.arange(kv_len)[None, :]
    cross = cross_mod(None, None, qi, ki)
    same = self_mod(None, None, qi, torch.arange(q_len)[None, :])
    assert torch.equal(cross, cross_visibility(kv_mask, ExpertCrossLayout(chunks, g, clean, noised)))
    assert torch.equal(same, self_visibility(q_mask, ExpertSelfLayout(chunks)))

    def sdpa(keys, vals, visibility):
        return F.scaled_dot_product_attention(
            q.transpose(0, 1)[None], keys.transpose(0, 1)[None], vals.transpose(0, 1)[None],
            attn_mask=visibility,
        )[0].transpose(0, 1)

    assert torch.allclose(actual_self, sdpa(qk, qv, same), atol=2e-5, rtol=1e-5)
    assert torch.allclose(actual_cross, sdpa(k, v, cross), atol=2e-5, rtol=1e-5)


@pytest.mark.parametrize("dummy_info_present", [False, True], ids=["replay", "capture"])
def test_capture_padding_has_no_request_layout_and_uses_zero_chunk_sentinel(
    dummy_info_present: bool,
) -> None:
    arrays = [
        Array(
            "expert_q", nodes={"expert"}, labels={"q"}, layers={0},
            entry=KVEntry(heads=2, head_dim=32), max_len=0,
            reads={"self_attn": Read(mask=ExpertSelf(2, 1), op=Attend(q_heads=2))},
        ),
        Array(
            "trunk_kv", nodes={"expert"}, labels={"kv"}, layers={0},
            entry=KVEntry(heads=2, head_dim=32), max_len=0,
            reads={"cross_attn": Read(
                mask=ExpertCross(2, 1, 1, 2), op=Attend(q_heads=2), queries="expert_q",
            )},
        ),
    ]
    resources = {spec.resource_key: SimpleNamespace(run=lambda *a, **kw: None) for spec in build_specs(arrays)}
    handles = node_handles(SimpleNamespace(declare_state=lambda: arrays), "expert", resources)
    info = CurrentForwardPassInfo(
        request_id="r0", graph_walk="chunk", fwd_index=0, random_seed=0, max_tokens=1,
        step_metadata={EXPERT_LAYOUT_KWARG: {
            "expert_q": {"q": {"self_attn": [1]}},
            "trunk_kv": {"kv": {"cross_attn": [1, 2, 0, 2]}},
        }},
    )
    step = SubmoduleStep(state={
        "expert_q": ArrayStep(segments=(Segment("r0", "q", 4), Segment(-2, "q", 4))),
        "trunk_kv": ArrayStep(segments=(Segment("r0", "kv", 4), Segment(-2, "kv", 5))),
    })
    infos = {"r0": info}
    if dummy_info_present:
        infos.update(dummy_metadata([-2], "chunk"))
    planned = state_steps(step, handles, per_request_info=infos)
    assert planned.steps["self_attn"].layouts == ((1, 0, 0, 2), (0, 4, 4, 0))
    assert planned.steps["cross_attn"].layouts == ((1, 2, 0, 2), (0, 4, 5, 0))


def test_two_request_cross_plan_keeps_labels_and_lengths_separate() -> None:
    from test.modular.test_ragged_expert_attention import _dense

    query = Array(
        "expert_q", nodes={"expert"}, labels={"q"}, layers={0},
        entry=KVEntry(heads=2, head_dim=32), max_len=0,
        reads={"self_attn": Read(mask=ExpertSelf(2, 1), op=Attend(q_heads=2))},
    )
    keys = Array(
        "trunk_kv", nodes={"expert"}, labels={"kv"}, layers={0},
        entry=KVEntry(heads=2, head_dim=32), max_len=0,
        reads={"cross_attn": Read(
            mask=ExpertCross(2, 1, 1, 2), op=Attend(q_heads=2), queries="expert_q",
        )},
    )
    arrays = [query, keys]
    specs = build_specs(arrays)
    resources = {
        spec.resource_key: build_resource(
            spec, EngineResourceInfo(device=torch.device("cpu"), kv_dtype=torch.float32),
        ) for spec in specs
    }
    handles = node_handles(SimpleNamespace(declare_state=lambda: arrays), "expert", resources)
    layouts = {"r0": (1, 2, 0, 2), "r1": (2, 2, 2, 4)}
    infos = {}
    for rid, (chunks, glob, clean, noised) in layouts.items():
        infos[rid] = CurrentForwardPassInfo(
            request_id=rid, graph_walk="chunk", fwd_index=0, random_seed=0, max_tokens=1,
            step_metadata={EXPERT_LAYOUT_KWARG: {
                "expert_q": {"q": {"self_attn": [chunks]}},
                "trunk_kv": {"kv": {"cross_attn": [chunks, glob, clean, noised]}},
            }},
        )
    step = SubmoduleStep(state={
        "expert_q": ArrayStep(segments=(Segment("r0", "q", 4), Segment("r1", "q", 8))),
        "trunk_kv": ArrayStep(segments=(Segment("r0", "kv", 4), Segment("r1", "kv", 8))),
    })
    planned = state_steps(step, handles, per_request_info=infos)
    ctx = StepContext(request_ids=["r0", "r1"], graph_walk="chunk", slot=0, capture=False)
    for spec in specs:
        resources[spec.resource_key].plan(planned.steps[spec.resource_key], ctx)
    assert planned.steps["cross_attn"].pairs == (("q", "kv"),)
    cross = _dense(resources["cross_attn"]._wrapper("q<-kv").block_mask, 12, 12)
    want = torch.block_diag(*(
        cross_visibility(ExpertCross(2, 1, 1, 2), ExpertCrossLayout(*values))
        for values in layouts.values()
    ))
    assert torch.equal(cross, want)


def test_missing_expert_layout_is_refused() -> None:
    array = Array(
        "expert_q", nodes={"expert"}, labels={"q"}, layers={0},
        entry=KVEntry(heads=2, head_dim=32), max_len=0,
        reads={"self_attn": Read(mask=ExpertSelf(2, 1), op=Attend(q_heads=2))},
    )
    handles = node_handles(
        SimpleNamespace(declare_state=lambda: [array]), "expert",
        {"self_attn": SimpleNamespace(run=lambda *a, **kw: None)},
    )
    info = CurrentForwardPassInfo(
        request_id="r0", graph_walk="chunk", fwd_index=0, random_seed=0, max_tokens=1,
    )
    step = SubmoduleStep(state={"expert_q": ArrayStep(segments=(Segment("r0", "q", 4),))})
    with pytest.raises(ValueError, match="has no expert state_layout"):
        state_steps(step, handles, per_request_info={"r0": info})
