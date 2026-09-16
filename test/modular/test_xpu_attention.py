"""CPU-side planning tests for the XPU paged-attention resource."""

import pytest
import torch

from mstar.engine.resources import (
    AttentionConfig,
    AttentionSpec,
    AttentionStep,
    AttnBackend,
    BucketKey,
    KVConfig,
    KVSpec,
    PositionConfig,
    PositionSpec,
    SlotLease,
    StepContext,
)
from mstar.engine.resources.attn import xpu as xpu_attention
from mstar.engine.resources.attn.base import AttentionManager
from mstar.engine.resources.attn.xpu import XPUPagedAttentionManager
from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.kv.plan import KVPlanOutput, SequenceView
from mstar.engine.resources.sampler.utils import _rng_offset_stride


def _kv_config() -> KVConfig:
    return KVConfig(
        num_layers=1,
        num_kv_heads=2,
        head_dim=8,
        max_seq_len=64,
        max_num_pages=16,
        page_size=4,
        num_qo_heads=4,
    )


def _manager() -> XPUPagedAttentionManager:
    return XPUPagedAttentionManager(
        kv_cache="kv",
        device=torch.device("cpu"),
        kv_config=_kv_config(),
    )


def _ctx(
    views: list[SequenceView],
    slot_lease: SlotLease | None = None,
) -> StepContext:
    return StepContext(
        request_ids=tuple(view.request_id for view in views),
        graph_walk="decode",
        slot=0,
        capture=False,
        slot_lease=slot_lease,
        plan_results={
            "kv": {
                "main": KVPlanOutput(
                    cpu_indptrs=None,
                    views=views,
                ),
            },
        },
    )


def _xpu_spec_and_info(device: str = "xpu"):
    kv_config = _kv_config()
    spec = AttentionSpec(
        resource_key="attn",
        nodes={"LLM"},
        config=AttentionConfig(
            kv_cache="kv",
            backend=AttnBackend.XPU_PAGED,
        ),
    )
    info = EngineResourceInfo(
        device=torch.device(device),
        dependencies={
            "kv": KVSpec(
                resource_key="kv",
                nodes={"LLM"},
                config=kv_config,
            ),
        },
    )
    return spec, info


def test_xpu_backend_selected_by_spec(monkeypatch):
    monkeypatch.setattr(
        xpu_attention, "_xpu_paged_unavailable_reason", lambda: None
    )
    spec, info = _xpu_spec_and_info()
    manager = AttentionManager.build(spec, info)
    assert isinstance(manager, XPUPagedAttentionManager)
    assert manager.depends_on() == {"kv"}


def test_xpu_backend_rejects_non_xpu_device():
    spec, info = _xpu_spec_and_info("cpu")
    with pytest.raises(RuntimeError, match="requires an XPU device"):
        AttentionManager.build(spec, info)


def test_xpu_backend_fails_fast_when_extension_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        xpu_attention,
        "_xpu_paged_unavailable_reason",
        lambda: "ImportError: missing extension",
    )
    spec, info = _xpu_spec_and_info()
    with pytest.raises(RuntimeError, match="vllm-xpu-kernels.*missing extension"):
        AttentionManager.build(spec, info)


def test_position_spec_declares_its_kv_dependency():
    spec = PositionSpec(
        resource_key="rope",
        nodes={"LLM"},
        config=PositionConfig(kv_cache="kv"),
    )
    assert spec.depends_on() == {"kv"}


def test_xpu_rng_offset_advances_past_the_full_logits_row():
    assert _rng_offset_stride("xpu", 152064) == 152064
    assert _rng_offset_stride("cuda", 152064) == 1


def test_xpu_plan_uses_kv_view_order_and_pads_block_table():
    views = [
        SequenceView("r0", "main", [2, 4], length=7, to_compute=3),
        SequenceView("r1", "main", [8], length=4, to_compute=1),
    ]
    manager = _manager()
    manager.plan(AttentionStep(causal=False), _ctx(views))

    plan = manager._current_plans["main"]
    assert plan.block_table.tolist() == [[2, 4], [8, 0]]
    assert plan.cu_q.tolist() == [0, 3, 4]
    assert plan.host_kv_lens.tolist() == [7, 4]
    assert plan.max_q == 3
    assert plan.max_k == 7
    assert plan.causal is False
    assert manager.qo_indptr_buf("main").tolist() == [0, 3, 4]


def test_xpu_graph_plan_keeps_addresses_stable_across_replans():
    manager = _manager()
    lease = SlotLease(
        slot=0,
        bucket=BucketKey(graph_walk="image_gen_cfg", bs=1, num_tokens=1),
    )
    manager.plan(
        AttentionStep(causal=False),
        _ctx(
            [SequenceView("r0", "main", [2], length=3, to_compute=1)],
            slot_lease=lease,
        ),
    )
    plan = manager._current_plans["main"]
    addresses = (
        plan.block_table.data_ptr(),
        plan.cu_q.data_ptr(),
        plan.host_kv_lens.data_ptr(),
    )

    manager.plan(
        AttentionStep(causal=False),
        _ctx(
            [
                SequenceView(
                    "r0", "main", [5, 6, 7], length=9, to_compute=1
                )
            ],
            slot_lease=lease,
        ),
    )

    replanned = manager._current_plans["main"]
    assert replanned is plan
    assert (
        replanned.block_table.data_ptr(),
        replanned.cu_q.data_ptr(),
        replanned.host_kv_lens.data_ptr(),
    ) == addresses
    assert replanned.block_table[0, :3].tolist() == [5, 6, 7]
    assert replanned.host_kv_lens.tolist() == [9]
    assert replanned.max_k == _kv_config().max_seq_len
