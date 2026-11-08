"""Planning and graph replay tests for the XPU paged-attention resource."""

import pytest
import torch

from mstar.engine.accelerator_graph_config import (
    BatchedAcceleratorGraphConfig,
    PackedAcceleratorGraphConfig,
    PiecewisePackedConfig,
)
from mstar.engine.resources import (
    AttentionConfig,
    AttentionSpec,
    AttentionStep,
    AttnBackend,
    BucketKey,
    KVSpec,
    PagedKVConfig,
    PositionConfig,
    PositionSpec,
    SlotLease,
    StepContext,
)
from mstar.engine.resources.attn import xpu as xpu_attention
from mstar.engine.resources.attn.base import AttentionManager
from mstar.engine.resources.attn.xpu import XPUPagedAttentionManager
from mstar.engine.resources.base import CGSlotSpec, EngineResourceInfo
from mstar.engine.resources.kv.plan import KVPlanOutput, SequenceView
from mstar.engine.resources.sampler.utils import _rng_offset_stride
from mstar.model.submodule_base import NodeInputs


def _kv_config() -> PagedKVConfig:
    return PagedKVConfig(
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
        bucket=BucketKey(graph_walk="decode", bs=1, num_tokens=1),
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


def test_xpu_graph_replan_stages_into_existing_buffers(monkeypatch):
    manager = _manager()
    lease = SlotLease(
        slot=0, bucket=BucketKey(graph_walk="decode", bs=1, num_tokens=1),
    )
    manager.plan(
        AttentionStep(causal=True),
        _ctx([SequenceView(1, "main", [2, 3], length=7, to_compute=1)], lease),
    )
    captured = manager._current_plans["main"]

    def unexpected_allocation(*args, **kwargs):
        pytest.fail("replanning must stage directly into the captured buffers")

    monkeypatch.setattr(torch, "tensor", unexpected_allocation)
    monkeypatch.setattr(torch, "empty", unexpected_allocation)
    ctx = _ctx([SequenceView(1, "main", [5], length=3, to_compute=1)], lease)
    ctx.is_preplan = True
    manager.plan(AttentionStep(causal=True), ctx)
    manager.plan(AttentionStep(causal=True), _ctx([], lease))

    assert manager._current_plans["main"] is captured
    assert captured.block_table.tolist() == [[5] + [0] * 15]
    assert captured.cu_q.tolist() == [0, 1]
    assert captured.host_kv_lens.tolist() == [3]


def test_xpu_graph_plan_supports_context_past_text_sequence_limit():
    config = _kv_config()
    config.max_num_pages = 32
    manager = XPUPagedAttentionManager(
        kv_cache="kv", device=torch.device("cpu"), kv_config=config,
    )
    lease = SlotLease(
        slot=0,
        bucket=BucketKey(graph_walk="decode", bs=1, num_tokens=1),
    )
    manager.plan(
        AttentionStep(causal=False),
        _ctx([SequenceView("r0", "main", [2], length=3, to_compute=1)], slot_lease=lease),
    )
    plan = manager._current_plans["main"]
    addresses = tuple(
        getattr(plan, name).data_ptr()
        for name in ("block_table", "cu_q", "host_kv_lens")
    )
    pages = list(range(1, 21))
    length = len(pages) * config.page_size
    assert length > config.max_seq_len
    manager.plan(
        AttentionStep(causal=False),
        _ctx([SequenceView("r0", "main", pages, length=length, to_compute=1)], slot_lease=lease),
    )
    assert manager._current_plans["main"] is plan
    assert tuple(
        getattr(plan, name).data_ptr()
        for name in ("block_table", "cu_q", "host_kv_lens")
    ) == addresses
    assert plan.block_table.shape == (1, config.max_num_pages)
    assert plan.block_table[0, :len(pages)].tolist() == pages
    assert not plan.block_table[0, len(pages):].any()
    assert plan.host_kv_lens.tolist() == [length]
    assert plan.max_k == config.max_num_pages * config.page_size


def _register_graph_bucket(manager, bucket, *, packed, piecewise=False):
    if piecewise:
        config = PiecewisePackedConfig(
            total_tokens=[bucket.num_tokens],
            capture_batch_sizes=[bucket.bs],
            capture_fn=lambda inp: {},
            make_static_inputs=lambda shape: {},
        )
    elif packed:
        config = PackedAcceleratorGraphConfig(
            capture_graph_walk=bucket.graph_walk,
            capture_token_lengths=[bucket.num_tokens],
            capture_batch_sizes=[bucket.bs],
            make_node_input=lambda n: NodeInputs(input_seq_len=n),
        )
    else:
        config = BatchedAcceleratorGraphConfig(
            capture_graph_walk=bucket.graph_walk,
            capture_batch_sizes=[bucket.bs],
            single_request_inputs=NodeInputs(input_seq_len=1),
        )
    manager.build_accelerator_graph_buffers(
        [CGSlotSpec(bucket=bucket, slot=0, config=config)],
        max_bs=bucket.bs,
        max_seq_len=bucket.num_tokens,
    )


@pytest.mark.parametrize(
    ("num_tokens", "capture_lengths", "replay_lengths"),
    [(16, [8, 8], [13, 3]), (2, [1, 1], [2, 0])],
)
@pytest.mark.parametrize("recipe", ["packed", "piecewise", "unregistered"])
def test_packed_graph_query_bound_covers_uneven_replans(
    num_tokens, capture_lengths, replay_lengths, recipe,
):
    manager = _manager()
    bucket = BucketKey(graph_walk="prefill_text", bs=2, num_tokens=num_tokens)
    if recipe != "unregistered":
        _register_graph_bucket(
            manager, bucket, packed=True, piecewise=recipe == "piecewise",
        )
    lease = SlotLease(slot=0, bucket=bucket)

    def plan_lengths(lengths):
        views = [
            SequenceView(
                rid, "main", list(range(1, (n + 3) // 4 + 1)),
                length=n, to_compute=n,
            )
            for rid, n in enumerate(lengths)
        ]
        manager.plan(AttentionStep(causal=True), _ctx(views, lease))

    plan_lengths(capture_lengths)
    captured = manager._current_plans["main"]
    addresses = tuple(
        getattr(captured, name).data_ptr()
        for name in ("block_table", "cu_q", "host_kv_lens")
    )
    assert captured.max_q == num_tokens
    plan_lengths(replay_lengths)
    replayed = manager._current_plans["main"]
    assert replayed is captured
    assert replayed.max_q == num_tokens
    assert replayed.cu_q.tolist() == [0, replay_lengths[0], sum(replay_lengths)]
    assert tuple(
        getattr(replayed, name).data_ptr()
        for name in ("block_table", "cu_q", "host_kv_lens")
    ) == addresses


def test_batched_decode_graph_keeps_single_token_query_bound():
    manager = _manager()
    bucket = BucketKey(graph_walk="decode", bs=2, num_tokens=2)
    _register_graph_bucket(manager, bucket, packed=False)
    views = [
        SequenceView(rid, "main", [1], length=3, to_compute=1)
        for rid in (1, 2)
    ]
    manager.plan(AttentionStep(causal=True), _ctx(views, SlotLease(slot=0, bucket=bucket)))
    assert manager._current_plans["main"].max_q == 1


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is unavailable")
def test_xpu_preplan_uploads_survive_slot_and_host_ring_reuse():
    device = torch.device("xpu:0")
    torch.xpu.set_device(device)
    manager = XPUPagedAttentionManager("kv", device, _kv_config())
    bucket = BucketKey(graph_walk="decode", bs=1, num_tokens=1)
    fields = ("block_table", "cu_q", "host_kv_lens")
    graphs, outputs, addresses = {}, {}, {}
    for slot in (0, 1):
        manager.plan(
            AttentionStep(causal=True),
            _ctx([SequenceView(1, "main", [1], length=3, to_compute=1)], SlotLease(slot=slot, bucket=bucket)),
        )
        plan = manager._current_plans["main"]
        addresses[slot] = tuple(getattr(plan, name).data_ptr() for name in fields)
        for name in fields:
            getattr(plan, name).clone()
        torch.xpu.synchronize()
        graphs[slot] = torch.xpu.XPUGraph()
        with torch.xpu.graph(graphs[slot]):
            outputs[slot] = tuple(getattr(plan, name).clone() for name in fields)

    execution_stream = torch.xpu.current_stream()
    plan_stream = torch.xpu.Stream()
    plan_stream.wait_stream(execution_stream)
    retired = {}
    history = []
    for iteration in range(20):
        slot = iteration % 2
        pages = list(range(1, iteration % 3 + 2))
        length = len(pages) * 4 - 1
        ctx = _ctx(
            [SequenceView(1, "main", pages, length=length, to_compute=1)],
            SlotLease(slot=slot, bucket=bucket),
        )
        ctx.is_preplan = True
        with torch.xpu.stream(plan_stream):
            if slot in retired:
                plan_stream.wait_event(retired[slot])
            manager.plan(AttentionStep(causal=True), ctx)
            ready = torch.xpu.Event()
            ready.record(plan_stream)
        execution_stream.wait_event(ready)
        ctx.is_preplan = False
        manager.plan(AttentionStep(causal=True), ctx)
        plan = manager._current_plans["main"]
        assert tuple(getattr(plan, name).data_ptr() for name in fields) == addresses[slot]
        graphs[slot].replay()
        history.append((
            tuple(tensor.clone() for tensor in outputs[slot]),
            ([[*pages, *([0] * (16 - len(pages)))]], [0, 1], [length]),
        ))
        retired[slot] = torch.xpu.Event()
        retired[slot].record(execution_stream)

    torch.xpu.synchronize()
    for actual, expected in history:
        for tensor, values in zip(actual, expected, strict=True):
            assert tensor.cpu().tolist() == values


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is unavailable")
def test_xpu_packed_attention_graph_replays_uneven_queries():
    reason = xpu_attention._xpu_paged_unavailable_reason()
    if reason is not None:
        pytest.skip(reason)
    device = torch.device("xpu:0")
    config = PagedKVConfig(
        num_layers=1, num_kv_heads=2, head_dim=64,
        max_seq_len=1024, max_num_pages=64, page_size=16, num_qo_heads=4,
    )
    manager = XPUPagedAttentionManager("kv", device, config)
    eager = XPUPagedAttentionManager("kv", device, config)
    bucket = BucketKey(graph_walk="prefill_text", bs=2, num_tokens=512)
    _register_graph_bucket(manager, bucket, packed=True)
    lease = SlotLease(slot=0, bucket=bucket)
    generator = torch.Generator().manual_seed(777)
    q = torch.randn(512, 4, 64, generator=generator).to(device, torch.bfloat16)
    cache = torch.randn(64, 2, 16, 2, 64, generator=generator).to(device, torch.bfloat16)

    def views_for(lengths):
        views = []
        next_page = 1
        for rid, length in enumerate(lengths):
            pages = (length + config.page_size - 1) // config.page_size
            views.append(SequenceView(
                rid, "main", list(range(next_page, next_page + pages)),
                length=length, to_compute=length,
            ))
            next_page += pages
        return views

    manager.plan(
        AttentionStep(causal=True), _ctx(views_for([256, 256]), lease),
    )
    manager.run(q, label="main", kv_cache_layer=cache)
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        captured_output = manager.run(q, label="main", kv_cache_layer=cache)

    for lengths in ([504, 8], [8, 504], [407, 5]):
        views = views_for(lengths)
        manager.plan(AttentionStep(causal=True), _ctx(views, lease))
        eager.plan(AttentionStep(causal=True), _ctx(views))
        total = sum(lengths)
        expected = eager.run(q[:total], label="main", kv_cache_layer=cache)
        captured_output.fill_(float("nan"))
        graph.replay()
        torch.xpu.synchronize()
        actual = captured_output[:total]
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
