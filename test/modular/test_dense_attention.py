"""Unit tests for the dense attention backend and the stream epoch it rides on.

The dense backend is the first resource to cache something *derived from the
contents* of a KV stream across steps. Nothing else in v1 does that, so the two
things worth pinning are the layout it builds off the KV plan, and the
invalidation: a cached gather must not survive a fork, a reset, or a page-table
move. Both run on CPU with no kernel — the FlashAttention-3 call itself is the
only part that needs a GPU.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources import (
    AttentionConfig,
    AttentionSpec,
    AttentionStep,
    AttnBackend,
    KVSpec,
    KVStep,
    PagedKVConfig,
    Segment,
    SlotLease,
    StepContext,
)
from mstar.engine.resources.attn import dense as dense_module
from mstar.engine.resources.attn.base import AttentionManager
from mstar.engine.resources.attn.dense import (
    DenseAttentionManager,
    _fa3_unavailable_reason,
)
from mstar.engine.resources.attn.flashinfer import FlashInferManager
from mstar.engine.resources.attn.ragged.dense_wrappers import DensePrefixedPrefillWrapper
from mstar.engine.resources.attn.ragged.wrappers import FlashInferPrefixedPrefillWrapper
from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.kv.plan import KVPlanOutput, SequenceView
from mstar.platforms import resolve_platform

PAGE_SIZE = 4
NUM_KV_HEADS = 2
HEAD_DIM = 3
MAX_PAGES = 8


def _kv_config() -> PagedKVConfig:
    return PagedKVConfig(
        num_layers=1,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        max_seq_len=64,
        max_num_pages=MAX_PAGES,
        page_size=PAGE_SIZE,
    )


def _manager(wrapper_class=DensePrefixedPrefillWrapper, **kwargs) -> DenseAttentionManager:
    return DenseAttentionManager(
        kv_cache="kv",
        device=torch.device("cpu"),
        dtype=torch.float32,
        kv_config=_kv_config(),
        wrapper_class=wrapper_class,
        **kwargs,
    )


def _ctx(views: list[SequenceView], slot: int = 0, **kwargs) -> StepContext:
    return StepContext(
        request_ids=tuple(dict.fromkeys(view.request_id for view in views)),
        graph_walk="gen",
        slot=slot,
        capture=False,
        plan_results={
            "kv": {"main": KVPlanOutput(cpu_indptrs=None, views=views)}
        },
        **kwargs,
    )


def _view(rid: str, pages: list[int], prefix: int, fresh: int):
    return SequenceView(
        request_id=rid,
        label="main",
        page_idxs=pages,
        length=prefix + fresh,
        to_compute=fresh,
    )


def _layer_tensor(fill: float = 0.0) -> torch.Tensor:
    """A KV layer whose every slot is a distinct number, so a gather that
    picked the wrong page or the wrong offset shows up as a value mismatch."""
    n = MAX_PAGES * 2 * PAGE_SIZE * NUM_KV_HEADS * HEAD_DIM
    return (torch.arange(n, dtype=torch.float32) + fill).reshape(
        MAX_PAGES, 2, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM
    )


class TestDensePlan:
    def test_prefix_is_what_the_step_does_not_compute(self):
        """The frozen prefix is `length - to_compute`: the fresh tokens are
        declared as ordinary spans (with commit=False), so the KV plan hands
        over both halves without the step saying anything backend-specific."""
        manager = _manager()
        # 6 resident + 5 fresh, over 3 pages of 4
        views = [_view("r0", [1, 2, 3], prefix=6, fresh=5)]
        manager.plan(AttentionStep(causal=False), _ctx(views))

        (segment,) = manager._current_plans["main"].segments
        assert segment.prefix_len == 6
        assert segment.q_len == 5
        # only the pages the prefix actually covers, not the ones reserved for
        # the fresh span
        assert list(segment.pages) == [1, 2]

    def test_cumulative_lengths_pack_in_kv_plan_order(self):
        manager = _manager()
        views = [
            _view("r0", [0, 1], prefix=3, fresh=2),
            _view("r1", [2, 3, 4], prefix=9, fresh=4),
        ]
        manager.plan(AttentionStep(causal=False), _ctx(views))
        plan = manager._current_plans["main"]

        assert list(plan.cu_q) == [0, 2, 6]
        assert list(plan.cu_k) == [0, 5, 18]
        assert plan.max_q == 4
        assert plan.max_k == 13
        assert [s.request_id for s in plan.segments] == ["r0", "r1"]
        assert plan.causal is False

    def test_only_the_labels_the_step_attends_are_planned(self):
        """A cfg rank's KV step holds both branches' labels and its attention reads only its own; a
        combined label is planned when its views carry an attended label."""
        manager = _manager()
        views = {
            "main": KVPlanOutput(cpu_indptrs=None, views=[_view("r0", [0], prefix=2, fresh=1)]),
            "uncond": KVPlanOutput(cpu_indptrs=None, views=[_view("r0", [1], prefix=2, fresh=1)._replace(
                label="uncond")]),
        }
        ctx = _ctx([])
        ctx.plan_results["kv"] = views
        manager.plan(AttentionStep(segments=(Segment("r0", "uncond", 1),), causal=False), ctx)
        assert manager._current_plans.keys() == {"uncond"}
        manager.plan(AttentionStep(causal=False), ctx)
        assert manager._current_plans.keys() == {"main", "uncond"}

    def test_causal_rides_on_the_step(self):
        manager = _manager()
        manager.plan(AttentionStep(causal=True), _ctx([_view("r0", [0], prefix=1, fresh=1)]))
        assert manager._current_plans["main"].causal is True

    def test_planning_under_a_graph_slot_is_rejected(self):
        """Eager-only: there is no captured replay of a gather whose shape the
        step decides, so a lease is a config error rather than a silent
        fall-through to a wrong path."""
        manager = _manager()
        ctx = _ctx([_view("r0", [0], prefix=2, fresh=2)])
        ctx.slot_lease = SlotLease(slot=0, bucket=None)
        with pytest.raises(RuntimeError, match="eager-only"):
            manager.plan(AttentionStep(causal=False), ctx)

    def test_missing_kv_plan_is_named(self):
        manager = _manager()
        ctx = _ctx([])
        ctx.plan_results.clear()
        with pytest.raises(AssertionError, match="expected plan result from kv"):
            manager.plan(AttentionStep(causal=False), ctx)


class TestPrefixGather:
    def _planned(self, manager, views):
        manager.plan(AttentionStep(causal=False), _ctx(views))
        return manager._current_plans["main"].segments

    def test_run_without_the_fresh_kv_is_an_error(self):
        manager = _manager()
        self._planned(manager, [_view("r0", [0], prefix=3, fresh=1)])
        with pytest.raises(AssertionError, match="dense attention runs on"):
            manager.run(torch.zeros(1, 1, 1), "main", _layer_tensor())


class TestBackendSelection:
    def test_dense_spec_builds_dense_when_fa3_is_there(self):
        """...and degrades to the paged backend when it is not: the two are
        drop-in for each other, so a missing wheel costs speed, not serving."""
        spec = AttentionSpec(
            resource_key="attn",
            nodes={"llm"},
            config=AttentionConfig(kv_cache="kv", backend=AttnBackend.DENSE),
        )
        assert spec.depends_on() == {"kv"}
        device = torch.device("cpu")
        manager = AttentionManager.build(
            spec,
            EngineResourceInfo(
                device=device,
                dependencies={
                    "kv": KVSpec(
                        resource_key="kv", nodes={"llm"}, config=_kv_config(),
                    ),
                },
            ),
        )
        expected = (
            DenseAttentionManager if _fa3_unavailable_reason(device) is None
            else FlashInferManager
        )
        assert isinstance(manager, expected)
        assert manager.depends_on() == {"kv"}

    def _build_on_cuda(self, monkeypatch, fa3_reason, flashinfer_reason, flashinfer_backend="auto"):
        """Build a dense spec for a CUDA device with FA3's and FlashInfer's
        availability forced; nothing runs, so this passes on CPU."""
        monkeypatch.setattr(dense_module, "workspace_bytes", lambda: 1024)
        monkeypatch.setattr(DenseAttentionManager, "_check_flashinfer_backend", lambda self: None)
        monkeypatch.setattr(dense_module, "_fa3_unavailable_reason", lambda device=None: fa3_reason)
        monkeypatch.setattr(
            dense_module, "_flashinfer_ragged_unavailable_reason", lambda device=None: flashinfer_reason,
        )
        spec = AttentionSpec(
            resource_key="attn_gen",
            nodes={"dit"},
            config=AttentionConfig(
                kv_cache="kv", backend=AttnBackend.DENSE, flashinfer_backend=flashinfer_backend,
            ),
        )
        return AttentionManager.build(
            spec,
            EngineResourceInfo(
                device=torch.device("cuda", 0),
                dependencies={"kv": KVSpec(resource_key="kv", nodes={"dit"}, config=PagedKVConfig(
                    num_layers=1, num_kv_heads=2, head_dim=128, max_seq_len=64, max_num_pages=MAX_PAGES,
                    page_size=PAGE_SIZE,
                ))},
            ),
        )

    def test_dense_without_fa3_runs_on_flashinfer_ragged(self, monkeypatch):
        """Where FA3 has no kernel (Blackwell), the dense pass keeps its
        semantics on FlashInfer's ragged prefill rather than going paged."""
        manager = self._build_on_cuda(
            monkeypatch, "fa3-fwd has no kernel for compute capability 10.3", None, "cudnn",
        )
        assert type(manager) is DenseAttentionManager
        assert manager._wrapper_class is FlashInferPrefixedPrefillWrapper
        assert manager._flashinfer_backend == "cudnn"
        assert manager.requires_kv_write is False
        assert manager.force_double_buffer is True

    def test_dense_with_fa3_holds_the_fa3_wrapper(self, monkeypatch):
        manager = self._build_on_cuda(monkeypatch, None, None, "cudnn")
        assert type(manager) is DenseAttentionManager
        assert manager._wrapper_class is DensePrefixedPrefillWrapper
        # FA3 has no FlashInfer backend to pick, and its plan stages nothing
        assert manager._flashinfer_backend is None
        assert manager.force_double_buffer is False

    def test_the_ragged_resources_and_dense_share_one_selection(self, monkeypatch):
        """Both go through ``dense_or_flashinfer``: patching FA3's reason moves them together."""
        from mstar.engine.resources.attn.ragged.base import dense_or_flashinfer

        for reason, want in ((None, "dense"), ("no kernel", "flashinfer")):
            monkeypatch.setattr(dense_module, "_fa3_unavailable_reason", lambda device=None, r=reason: r)
            assert dense_or_flashinfer(torch.device("cuda", 0), "dense", "flashinfer", "Test") == want

    def test_dense_without_fa3_or_flashinfer_goes_paged(self, monkeypatch, caplog):
        import logging

        dense_module._warn_dense_fallback.cache_clear()
        try:
            with caplog.at_level(logging.WARNING, logger=dense_module.logger.name):
                manager = self._build_on_cuda(monkeypatch, "no fa3", "ImportError: no flashinfer")
        finally:
            dense_module._warn_dense_fallback.cache_clear()
        assert type(manager) is FlashInferManager
        assert "no fa3; ImportError: no flashinfer" in caplog.text

    def test_dense_kernels_do_not_share_a_fingerprint(self):
        """They differ in the last bits, so a cached artifact keyed on the
        resource must not carry over from one to the other."""
        prints = {
            _manager().fingerprint(),
            _manager(FlashInferPrefixedPrefillWrapper).fingerprint(),
            _manager(FlashInferPrefixedPrefillWrapper, flashinfer_backend="cudnn").fingerprint(),
        }
        assert len(prints) == 3


class _StubFlashInferWrapper(FlashInferPrefixedPrefillWrapper):
    """Stands in for FlashInfer's wrapper: refuses every backend but ``auto``."""

    def __init__(self, workspace_buffer=None, *, backend="auto", **kwargs):
        del workspace_buffer, kwargs
        self.backend = backend

    def plan(self, q_cu_seqlens, kv_cu_seqlens, causal=False):
        if self.backend != "auto":
            raise RuntimeError(f"no {self.backend} kernel for this GPU")

    def run(self, q, k, v):
        return torch.zeros_like(q)


class TestFlashInferBackendCheck:
    """A backend the GPU or the FlashInfer build cannot run is refused when
    the resource is built, not on every request."""

    @pytest.mark.parametrize("backend", ["cudnnn", "cutlass"])
    def test_a_backend_that_cannot_run_raises_naming_it(self, backend):
        manager = _manager(_StubFlashInferWrapper, flashinfer_backend=backend)
        with pytest.raises(ValueError, match=f"flashinfer_backend='{backend}' cannot run the dense attention"):
            manager._check_flashinfer_backend()

    def test_a_backend_that_runs_passes(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
        _manager(_StubFlashInferWrapper, flashinfer_backend="auto")._check_flashinfer_backend()

    def test_the_check_runs_when_the_manager_is_built_on_cuda(self, monkeypatch):
        calls = []
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(dense_module, "workspace_bytes", lambda: 0)
        monkeypatch.setattr(dense_module.torch, "empty", lambda *a, **k: None)


        def check(manager):
            calls.append(manager)

        monkeypatch.setattr(DenseAttentionManager, "_check_flashinfer_backend", check)
        manager = DenseAttentionManager(
            kv_cache="kv", device=torch.device("cuda", 0), dtype=torch.bfloat16, kv_config=_kv_config(),
            wrapper_class=_StubFlashInferWrapper, flashinfer_backend="cudnnn",
        )
        assert calls == [manager]

    def test_only_the_dense_backend_skips_the_kv_write(self):
        assert _manager().requires_kv_write is False
        assert FlashInferManager(
            kv_cache="kv",
            device=torch.device("cpu"),
            dtype=torch.float32,
            kv_config=_kv_config(),
        ).requires_kv_write is True


def _gpu_flashinfer():
    if not torch.cuda.is_available():
        return False
    try:
        import flashinfer  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


@pytest.mark.skipif(not _gpu_flashinfer(), reason="needs a CUDA GPU and flashinfer")
@pytest.mark.parametrize("backend", ["cudnnn", "not-a-backend"])
def test_flashinfer_backend_the_gpu_cannot_run_fails_at_build(backend):
    with pytest.raises(ValueError, match=f"flashinfer_backend='{backend}'"):
        DenseAttentionManager(
            kv_cache="kv", device=torch.device("cuda"), dtype=torch.bfloat16,
            kv_config=PagedKVConfig(
                num_layers=1, num_kv_heads=2, head_dim=128, max_seq_len=256, max_num_pages=16,
                page_size=16, num_qo_heads=4,
            ),
            wrapper_class=FlashInferPrefixedPrefillWrapper, flashinfer_backend=backend,
        )


@pytest.mark.parametrize("head_dim, offset", [(128, 0), (128, 1), (3, 0)], ids=["int64", "unaligned", "odd-dim"])
@pytest.mark.parametrize("contiguous", [True, False], ids=["one-block", "strided-layer"])
def test_each_layer_gathers_the_prefix_beside_the_fresh_kv_and_holds_nothing(monkeypatch, head_dim, offset, contiguous):
    """Every run assembles [prefix | fresh] per segment straight from the pages: the bytes of the prefix rows and
    the fresh rows concatenated, with a strided V (a slice of a fused QKV row) and a layer that isn't one block.
    Nothing is held between steps (a copy of every layer's prefix held for a whole denoise was GiBs per request),
    so a step after the pages changed reads the new pages."""
    page, hkv = 4, 2
    cfg = PagedKVConfig(
        num_layers=1, num_kv_heads=hkv, head_dim=head_dim, max_seq_len=64, max_num_pages=8, page_size=page,
    )
    manager = DenseAttentionManager(
        kv_cache="kv", device=torch.device("cpu"), dtype=torch.bfloat16, kv_config=cfg,
        wrapper_class=DensePrefixedPrefillWrapper,
    )
    seen = []
    monkeypatch.setattr(
        DensePrefixedPrefillWrapper, "run", lambda self, q, k, v, return_lse=False: seen.append((k, v)) or q,
    )
    layer = torch.randn(8, 2, page, hkv, head_dim + (not contiguous), dtype=torch.bfloat16)[..., :head_dim]
    segs = [("r0", [5, 2], 6, 3), ("r1", [], 0, 2), ("r2", [7], 3, 4)]  # rid, prefix pages, prefix, fresh
    total = sum(fr for *_, fr in segs)
    width = hkv * head_dim
    qkv = torch.randn(total, 3 * width + offset, dtype=torch.bfloat16)
    k = qkv[:, offset:offset + width].reshape(total, hkv, head_dim)
    v = qkv[:, offset + 2 * width:offset + 3 * width].reshape(total, hkv, head_dim)
    for _ in range(2):
        manager.plan(AttentionStep(causal=False), _ctx([
            _view(rid, pages + [0], prefix=pre, fresh=fr) for rid, pages, pre, fr in segs
        ]))
        manager.run(torch.zeros(total, 2, head_dim), "main", layer, k=k, v=v, layer_idx=0)
        key, val = seen.pop()
        off, want_k, want_v = 0, [], []
        for _, pages, pre, fr in segs:
            rows = layer[pages]
            want_k += [rows[:, 0].reshape(-1, hkv, head_dim)[:pre], k[off:off + fr]]
            want_v += [rows[:, 1].reshape(-1, hkv, head_dim)[:pre], v[off:off + fr]]
            off += fr
        assert torch.equal(key, torch.cat(want_k)) and torch.equal(val, torch.cat(want_v))
        layer.mul_(-1)


@pytest.mark.parametrize("wrapper, head_dim, extra_rows, capability, fixed", [
    ("prefixed", 128, 0, (9, 0), True),
    ("prefixed", 128, 2, (9, 0), False),  # rows past the plan's: the fixed-length kernel would attend them
    ("prefixed", 72, 0, (9, 0), False),
    ("ragged", 128, 0, (9, 0), False),  # cacheless, as Cosmos3's vision tower
    ("prefixed", 128, 0, (10, 0), False),  # proven on sm_90 only
    ("prefixed", 128, 0, None, False),  # no NVIDIA GPU
], ids=["proven", "past-the-plan", "head_dim-72", "cacheless", "sm_100", "no-gpu"])
def test_one_segment_takes_the_fixed_length_kernel_only_where_proven(
    monkeypatch, wrapper, head_dim, extra_rows, capability, fixed,
):
    """FA3's fixed-length kernel is proven bit-equal to varlen for one segment on the prefixed wrapper at
    head_dim 128 (Qwen-Image-2.1) on sm_90; every other one-segment plan stays on varlen."""
    from mstar.engine.resources.attn.ragged import dense_wrappers
    from mstar.engine.resources.attn.ragged.dense_wrappers import DenseRaggedPrefillWrapper

    called = []
    monkeypatch.setattr(resolve_platform(), "cuda_capability", lambda device=None: capability)
    monkeypatch.setattr(dense_wrappers, "flash_attn_func", lambda q, *a, **kw: called.append("fixed") or q)
    monkeypatch.setattr(dense_wrappers, "flash_attn_varlen_func", lambda q, *a, **kw: called.append("varlen") or q)
    cls = DensePrefixedPrefillWrapper if wrapper == "prefixed" else DenseRaggedPrefillWrapper
    w = cls(num_qo_heads=2, num_kv_heads=2, head_dim=head_dim, device=torch.device("cpu"), q_data_type=torch.float32)
    q_len, k_len = 5, 5 if wrapper == "ragged" else 9
    if wrapper == "prefixed":
        w.plan(torch.tensor([0, q_len], dtype=torch.int32), torch.tensor([0, k_len], dtype=torch.int32))
    else:
        w.plan(torch.tensor([0, q_len], dtype=torch.int32))
    q = torch.zeros(q_len + extra_rows, 2, head_dim)
    k = torch.zeros(k_len + extra_rows, 2, head_dim)
    w.run(q, k, k)
    assert called == ["fixed" if fixed else "varlen"]


def _wrapper_classes():
    classes = []
    if _gpu_flashinfer():
        classes.append(pytest.param(FlashInferPrefixedPrefillWrapper, id="flashinfer"))
    if torch.cuda.is_available() and _fa3_unavailable_reason(torch.device("cuda")) is None:
        classes.append(pytest.param(DensePrefixedPrefillWrapper, id="fa3"))
    return classes or [pytest.param(None, marks=pytest.mark.skip(reason="needs a CUDA GPU with FA3 or flashinfer"))]


@pytest.mark.parametrize("wrapper_class", _wrapper_classes())
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("segs", [
    [("r0", [3, 4], 20, 37), ("r1", [7], 9, 50)],  # rid, prefix pages, prefix, fresh
    [("r0", [3, 4], 20, 37)],
], ids=["two", "one"])
def test_dense_matches_sdpa_over_prefix_and_fresh(wrapper_class, causal, segs):
    """Segments with different frozen prefixes and GQA: each query block
    attends its own [prefix | fresh] keys, and a causal plan aligns the block
    to the end of them, on either kernel. One segment on FA3 takes its
    fixed-length kernel, with the varlen kernel's bits."""
    dev = torch.device("cuda")
    page, hq, hkv, d = 16, 4, 2, 128
    cfg = PagedKVConfig(
        num_layers=1, num_kv_heads=hkv, head_dim=d, max_seq_len=256, max_num_pages=16,
        page_size=page, num_qo_heads=hq,
    )
    manager = DenseAttentionManager(
        kv_cache="kv", device=dev, dtype=torch.bfloat16, kv_config=cfg, wrapper_class=wrapper_class,
    )
    torch.manual_seed(0)
    layer = torch.randn(16, 2, page, hkv, d, device=dev, dtype=torch.bfloat16)
    views = [_view(rid, pages + [15], prefix=pre, fresh=fr) for rid, pages, pre, fr in segs]
    manager.plan(AttentionStep(causal=causal), _ctx(views))
    total = sum(fr for *_, fr in segs)
    q = torch.randn(total, hq, d, device=dev, dtype=torch.bfloat16)
    k = torch.randn(total, hkv, d, device=dev, dtype=torch.bfloat16)
    v = torch.randn(total, hkv, d, device=dev, dtype=torch.bfloat16)
    out = manager.run(q, "main", layer, k=k, v=v, layer_idx=0)
    if wrapper_class is DensePrefixedPrefillWrapper and len(segs) == 1:
        (_, pages, pre, fr), = segs
        rows = layer[pages]
        kk = torch.cat([rows[:, 0].reshape(-1, hkv, d)[:pre], k])
        vv = torch.cat([rows[:, 1].reshape(-1, hkv, d)[:pre], v])
        cu_q = torch.tensor([0, fr], dtype=torch.int32, device=dev)
        cu_k = torch.tensor([0, pre + fr], dtype=torch.int32, device=dev)
        varlen = dense_module.flash_attn_varlen_func(q, kk, vv, cu_q, cu_k, fr, pre + fr, causal=causal,
                                        softmax_scale=d ** -0.5)
        assert torch.equal(out, varlen[0] if isinstance(varlen, tuple) else varlen)

    off = 0
    for _, pages, pre, fr in segs:
        rows = layer[pages]
        kk = torch.cat([rows[:, 0].reshape(-1, hkv, d)[:pre], k[off:off + fr]])
        vv = torch.cat([rows[:, 1].reshape(-1, hkv, d)[:pre], v[off:off + fr]])
        mask = None
        if causal:
            # query i sees the prefix and fresh tokens 0..i
            mask = torch.ones(fr, pre + fr, dtype=torch.bool, device=dev).tril(diagonal=pre)
        ref = torch.nn.functional.scaled_dot_product_attention(
            q[off:off + fr].transpose(0, 1).float(), kk.transpose(0, 1).float(), vv.transpose(0, 1).float(),
            attn_mask=mask, enable_gqa=True,
        ).transpose(0, 1)
        torch.testing.assert_close(out[off:off + fr].float(), ref, atol=2e-2, rtol=2e-2)
        off += fr


@pytest.fixture
def kv_manager(monkeypatch):
    """A KVManager over a CPU cache. Its transfer engine registers the cache
    for CUDA IPC at construction, which needs a GPU and has nothing to do with
    page bookkeeping, so it is stubbed out."""
    from mstar.engine.resources.kv import manager as kv_manager_module

    monkeypatch.setattr(
        kv_manager_module,
        "KVTransferManager",
        lambda info, kv_cache, **kwargs: None,
    )
    return kv_manager_module.KVManager(
        cfg=_kv_config(),
        name="kv",
        joint_comm_group=None,
        transfer_engine_info=None,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


class TestStreamGeneration:
    """`CacheStream.generation` is read through `SequenceView.generation` (and by the
    offload claim check), so its contract is worth pinning here: it moves when the resident bytes may have moved
    under that reader, and stands still when they cannot have."""

    def _plan_views(self, kv, segments, **step_kwargs):
        step = KVStep(segments=tuple(segments), **step_kwargs)
        ctx = StepContext(
            request_ids=tuple(s.request_id for s in segments),
            graph_walk="gen", slot=0, capture=False,
        )
        assert kv.admit(step, ctx).ok
        out = kv.plan(step, ctx)
        kv.commit(step, ctx)
        return out, step, ctx

    def test_denoise_steps_do_not_invalidate_a_resident_prefix(self, kv_manager):
        # the dense backend's only consumer (cosmos3) drives it with
        # commit=False: each denoise step recomputes its fresh K/V in place
        # rather than appending. commit is what moves the epoch (it is how an
        # in-place page write blocks an offload), so a commit=False step within
        # the pages already held leaves the epoch where it was, so the offload
        # claim check, its other reader, sees no write behind it. (Each layer
        # gathers the prefix from the pages every step regardless.)
        kv = kv_manager
        kv.ingest_request("r0")
        # commit the frozen prefix once, then read the epoch the denoise steps
        # ride on (after the prefix's own commit has moved it)
        self._plan_views(kv, [Segment("r0", "main", 6)])
        out, _, _ = self._plan_views(kv, [Segment("r0", "main", 1)], commit=False)
        first = out["main"].views[0].generation

        out, _, _ = self._plan_views(kv, [Segment("r0", "main", 1)], commit=False)
        assert out["main"].views[0].generation == first

    def test_taking_new_pages_moves_the_epoch(self, kv_manager):
        kv = kv_manager
        kv.ingest_request("r0")
        out, _, _ = self._plan_views(kv, [Segment("r0", "main", 4)])
        first = out["main"].views[0].generation

        out, _, _ = self._plan_views(kv, [Segment("r0", "main", 4)])
        assert out["main"].views[0].generation > first

    def test_a_fork_moves_the_target_epoch(self, kv_manager):
        """The case a page-identity fingerprint would miss: the fork copies
        over the target's existing pages, so nothing about the page list or
        the length changes — only the contents."""
        kv = kv_manager
        kv.ingest_request("r0")
        self._plan_views(kv, [Segment("r0", "main", 4)])
        # give the target a stream of its own at the same length
        self._plan_views(kv, [Segment("r0", "cfg", 4)])
        out, _, _ = self._plan_views(kv, [Segment("r0", "cfg", 0)])
        before = out["cfg"].views[0]
        # a view holds the stream's page list by reference, so snapshot it
        before_pages, before_len = list(before.page_idxs), before.length

        self._plan_views(kv, [Segment("r0", "cfg", 0)], post_forks=(("main", "cfg"),))
        out, _, _ = self._plan_views(kv, [Segment("r0", "cfg", 0)])
        after = out["cfg"].views[0]
        assert list(after.page_idxs) == before_pages
        assert after.length == before_len
        assert after.generation > before.generation

    def test_a_reset_moves_the_epoch(self, kv_manager):
        kv = kv_manager
        kv.ingest_request("r0")
        out, _, _ = self._plan_views(kv, [Segment("r0", "main", 4)])
        before = out["main"].views[0].generation

        kv.reset_request("r0")
        out, _, _ = self._plan_views(kv, [Segment("r0", "main", 4)])
        assert out["main"].views[0].generation > before


@pytest.mark.parametrize("batched", [True, False], ids=["batched-cfg", "plan-per-branch"])
def test_cosmos3s_eager_denoise_attends_each_branch_prefix_from_the_pages(kv_manager, monkeypatch, batched):
    """Cosmos3's eager denoise on the dense backend (``attention_backend="dense_gen"``): its own step declaration
    (cond and uncond in one combined plan, or a plan each), planned by the KV manager over committed text
    prefixes, attends every branch's [prefix | fresh] as the prefix commit wrote it, at every step."""
    from mstar.model.cosmos3.submodules import (
        CFG_BATCHED_LABEL,
        COND_LABEL,
        KV_CACHE,
        UNCOND_LABEL,
        Cosmos3DiTSubmodule,
    )

    kv, rids, fresh = kv_manager, ("r0", "r1"), 2
    prefix = {COND_LABEL: 5, UNCOND_LABEL: 2}
    for rid in rids:
        kv.ingest_request(rid)
    out, _, _ = TestStreamGeneration()._plan_views(
        kv, [Segment(rid, label, n) for label, n in prefix.items() for rid in rids],
    )
    # the prefix's K/V as its layers would have written it: a distinct value per (request, label, token)
    tensor, written = kv.kv_cache.tensor, {}
    for label, plan in out.items():
        for view in plan.views:
            for t in range(prefix[label]):
                value = 100 * rids.index(view.request_id) + 10 * (label == UNCOND_LABEL) + t + 1
                page, slot = view.page_idxs[t // PAGE_SIZE], t % PAGE_SIZE
                tensor[0, page, 0, slot], tensor[0, page, 1, slot] = value, -value
                written[view.request_id, label, t] = value

    manager = DenseAttentionManager(
        kv_cache="kv", device=torch.device("cpu"), dtype=torch.float32, kv_config=_kv_config(),
        wrapper_class=DensePrefixedPrefillWrapper,
    )
    seen = []
    monkeypatch.setattr(
        DensePrefixedPrefillWrapper, "run", lambda self, q, k, v, return_lse=False: seen.append((k, v)) or q,
    )
    step = Cosmos3DiTSubmodule._gen_step(
        Cosmos3DiTSubmodule, list(rids), (fresh,) * len(rids), (COND_LABEL, UNCOND_LABEL), combined=batched,
        attn_key="attn",
    )
    kv_step = step.steps[KV_CACHE]
    for _ in range(2):
        ctx = StepContext(request_ids=rids, graph_walk="gen", slot=0, capture=False)
        assert kv.admit(kv_step, ctx).ok
        ctx.plan_results["kv"] = kv.plan(kv_step, ctx)
        kv.commit(kv_step, ctx)
        manager.plan(step.steps["attn"], ctx)
        # label-major, as a combined plan concatenates its sources and Cosmos3 packs its rows
        branches = {label: [(rid, label) for rid in rids] for label in prefix}
        assert {label: [(s.request_id, s.label) for s in plan.segments]
                for label, plan in manager._current_plans.items()} == (
            {CFG_BATCHED_LABEL: [*branches[COND_LABEL], *branches[UNCOND_LABEL]]} if batched else branches)
        for label, plan in manager._current_plans.items():
            n = sum(s.q_len for s in plan.segments)
            k = torch.full((n, NUM_KV_HEADS, HEAD_DIM), 0.5)
            manager.run(torch.zeros(n, 2, HEAD_DIM), label, tensor[0], k=k, v=-k, layer_idx=0)
            key, val = seen.pop()
            want = []
            for segment in plan.segments:
                want += [written[segment.request_id, segment.label, t] for t in range(prefix[segment.label])]
                want += [0.5] * fresh
            assert key[:, 0, 0].tolist() == want and val[:, 0, 0].tolist() == [-w for w in want]
