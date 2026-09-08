"""The MLA attention backend and the latent KV layout it reads, on CPU.

Pins (1) the ``KVLayout.MLA`` cache: shape, the single-latent write/read,
page copies and chunk views; (2) the manager's plan: ``kv_len_arr`` derived
from the KV views, last-token indices for prefill sampling; (3) the fp32 SDPA
fallback against a naive dense reference — the contract the FlashInfer kernel
path is held to on sm90 (the kernel itself is GPU-only). (4) the spec builds
the right backend and validates its config.
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
    KVConfig,
    KVSpec,
    StepContext,
)
from mstar.engine.resources.attn.base import AttentionManager
from mstar.engine.resources.attn.mla import MLAAttentionManager
from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.kv.cache import KVCache
from mstar.engine.resources.kv.config import KVLayout
from mstar.engine.resources.kv.plan import (
    KVPlanOutput,
    KVPlanOutputs,
    SequenceView,
    build_paged_indptrs,
)

PAGE_SIZE = 4
CKV = 6
KPE = 2
HEAD_DIM = CKV + KPE
NUM_HEADS = 3
MAX_PAGES = 8
SCALE = 0.37


def _kv_config(**overrides) -> KVConfig:
    kwargs = dict(
        num_layers=2,
        num_kv_heads=1,
        head_dim=HEAD_DIM,
        max_seq_len=64,
        max_num_pages=MAX_PAGES,
        page_size=PAGE_SIZE,
        num_qo_heads=NUM_HEADS,
        layout=KVLayout.MLA,
    )
    kwargs.update(overrides)
    return KVConfig(**kwargs)


def _manager() -> MLAAttentionManager:
    return MLAAttentionManager(
        kv_cache="kv",
        device=torch.device("cpu"),
        dtype=torch.float32,
        kv_config=_kv_config(),
        ckv_dim=CKV,
        softmax_scale=SCALE,
    )


def _view(rid: str, pages: list[int], prefix: int, fresh: int) -> SequenceView:
    return SequenceView(
        request_id=rid, label="main", page_idxs=pages,
        length=prefix + fresh, to_compute=fresh,
    )


def _planned(manager: MLAAttentionManager, views: list[SequenceView]) -> StepContext:
    out = KVPlanOutput(
        cpu_indptrs=build_paged_indptrs(views, PAGE_SIZE), views=views,
    )
    ctx = StepContext(
        request_ids=tuple(v.request_id for v in views),
        graph_walk="gen", slot=0, capture=False,
        plan_results={"kv": KVPlanOutputs({"main": out})},
    )
    manager.plan(AttentionStep(causal=True), ctx)
    return ctx


def _reference(q: torch.Tensor, latent: torch.Tensor, old_len: int) -> torch.Tensor:
    """Dense causal MLA for one request: q [sl, H, ckv+kpe], latent [total, ckv+kpe]."""
    sl = q.shape[0]
    total = latent.shape[0]
    scores = torch.einsum("qhd,kd->hqk", q.float(), latent.float()) * SCALE
    q_pos = old_len + torch.arange(sl)
    causal = torch.arange(total)[None, :] <= q_pos[:, None]
    scores = scores.masked_fill(~causal, float("-inf")).softmax(-1)
    return torch.einsum("hqk,kd->qhd", scores, latent[:, :CKV].float())


class TestMlaCacheLayout:
    def test_shape_and_single_latent_write(self):
        cache = KVCache(_kv_config(), torch.device("cpu"), torch.float32)
        assert tuple(cache.tensor.shape) == (2, MAX_PAGES, PAGE_SIZE, HEAD_DIM)
        assert tuple(cache.layer_view(1).shape) == (MAX_PAGES, PAGE_SIZE, HEAD_DIM)

        latent = torch.arange(3 * HEAD_DIM, dtype=torch.float32).reshape(3, HEAD_DIM)
        page_idx = torch.tensor([2, 2, 5])
        cache_idx = torch.tensor([0, 1, 3])
        cache.write_tokens(1, latent, None, page_idx, cache_idx)
        assert torch.equal(cache.layer_view(1)[2, 0], latent[0])
        assert torch.equal(cache.layer_view(1)[2, 1], latent[1])
        assert torch.equal(cache.layer_view(1)[5, 3], latent[2])
        assert torch.equal(cache.read_tokens(1, page_idx, cache_idx), latent)
        # the other layer is untouched
        assert cache.layer_view(0).abs().sum() == 0

    def test_mla_refuses_a_value_tensor(self):
        cache = KVCache(_kv_config(), torch.device("cpu"), torch.float32)
        with pytest.raises(ValueError, match="v=None"):
            cache.write_tokens(
                0, torch.zeros(1, HEAD_DIM), torch.zeros(1, HEAD_DIM),
                torch.tensor([1]), torch.tensor([0]),
            )

    def test_copy_pages_and_chunk_view(self):
        cache = KVCache(_kv_config(), torch.device("cpu"), torch.float32)
        cache.tensor[:, 3] = 7.0
        cache.copy_pages([3], [6])
        assert torch.equal(cache.tensor[:, 6], cache.tensor[:, 3])
        chunk = cache.chunk_view(1, 6, 1, 3)
        assert tuple(chunk.shape) == (2, HEAD_DIM)
        ptrs, nbytes = cache.chunk_ptrs(1, 6, 1, 3)
        assert len(ptrs) == 1
        assert nbytes == 2 * HEAD_DIM * cache.tensor.element_size()
        expected = cache.data_ptr() + (
            1 * cache.tensor.stride(0) + 6 * cache.tensor.stride(1) + 1 * cache.tensor.stride(2)
        ) * cache.tensor.element_size()
        assert ptrs[0] == expected

    def test_layout_needs_one_kv_head(self):
        with pytest.raises(ValueError, match="one latent head"):
            _kv_config(num_kv_heads=2)

    def test_shard_keeps_the_latent_and_splits_query_heads(self):
        cfg = _kv_config(num_qo_heads=12)
        cfg.shard(4)
        assert cfg.num_kv_heads == 1
        assert cfg.num_qo_heads == 3


class TestMlaPlan:
    def test_kv_len_is_the_total_resident_length(self):
        views = [_view("a", [1, 2], prefix=5, fresh=2), _view("b", [3], prefix=0, fresh=3)]
        out = KVPlanOutput(cpu_indptrs=build_paged_indptrs(views, PAGE_SIZE), views=views)
        assert MLAAttentionManager.kv_len_arr(out).tolist() == [7, 3]
        # and agrees with what the paged indptrs say
        ind = out.cpu_indptrs
        derived = [
            (ind.paged_kv_indptr[i + 1] - ind.paged_kv_indptr[i] - 1).item() * PAGE_SIZE
            + ind.paged_kv_last_page_len[i].item()
            for i in range(2)
        ]
        assert derived == [7, 3]

    def test_select_last_hidden_follows_plan_order(self):
        manager = _manager()
        _planned(manager, [_view("a", [1], prefix=0, fresh=3), _view("b", [2], prefix=0, fresh=2)])
        hidden = torch.arange(5, dtype=torch.float32)[:, None].expand(5, 4)
        last = manager.select_last_hidden(hidden)
        assert last[:, 0].tolist() == [2.0, 4.0]

    def test_fallback_is_the_cpu_path(self):
        manager = _manager()
        assert manager.use_kernel is False
        assert manager.requires_kv_write is True
        assert manager.depends_on() == {"kv"}


class TestMlaFallbackParity:
    def test_mixed_prefill_and_decode_batch_matches_dense_reference(self):
        torch.manual_seed(0)
        cache = KVCache(_kv_config(), torch.device("cpu"), torch.float32)
        layer = cache.layer_view(0)
        # request a: 5 resident + 2 new (prefill continue); b: 0 + 3 (first
        # prefill); c: 4 resident + 1 new (decode). Pages are deliberately
        # non-contiguous so a gather that ignored the page table would show.
        views = [
            _view("a", [6, 1], prefix=5, fresh=2),
            _view("b", [3], prefix=0, fresh=3),
            _view("c", [7, 2], prefix=4, fresh=1),
        ]
        latents = {}
        for view in views:
            total = view.length
            rows = torch.randn(total, HEAD_DIM)
            latents[view.request_id] = rows
            for t in range(total):
                page = view.page_idxs[t // PAGE_SIZE]
                layer[page, t % PAGE_SIZE] = rows[t]

        manager = _manager()
        _planned(manager, views)
        T = sum(v.to_compute for v in views)
        q_nope = torch.randn(T, NUM_HEADS, CKV)
        q_pe = torch.randn(T, NUM_HEADS, KPE)
        out = manager.run_mla(q_nope, q_pe, label="main", kv_cache_layer=layer)
        assert tuple(out.shape) == (T, NUM_HEADS, CKV)

        q = torch.cat([q_nope, q_pe], dim=-1)
        start = 0
        for view in views:
            sl = view.to_compute
            ref = _reference(q[start:start + sl], latents[view.request_id], view.length - sl)
            torch.testing.assert_close(out[start:start + sl], ref, atol=1e-5, rtol=1e-5)
            start += sl

    def test_run_splits_a_concatenated_query(self):
        torch.manual_seed(1)
        cache = KVCache(_kv_config(), torch.device("cpu"), torch.float32)
        layer = cache.layer_view(0)
        layer[1, :3] = torch.randn(3, HEAD_DIM)
        manager = _manager()
        _planned(manager, [_view("a", [1], prefix=0, fresh=3)])
        q_nope = torch.randn(3, NUM_HEADS, CKV)
        q_pe = torch.randn(3, NUM_HEADS, KPE)
        via_run = manager.run(torch.cat([q_nope, q_pe], -1), "main", layer)
        via_mla = manager.run_mla(q_nope, q_pe, "main", layer)
        torch.testing.assert_close(via_run, via_mla)

    def test_zero_pe_is_exact_nope(self):
        """A NoPE model zero-pads kpe to the kernel's width: the pe term
        must contribute exactly nothing."""
        torch.manual_seed(2)
        cache = KVCache(_kv_config(), torch.device("cpu"), torch.float32)
        layer = cache.layer_view(0)
        rows = torch.randn(4, HEAD_DIM)
        rows[:, CKV:] = 0
        layer[2, :4] = rows
        manager = _manager()
        _planned(manager, [_view("a", [2], prefix=0, fresh=4)])
        q_nope = torch.randn(4, NUM_HEADS, CKV)
        out = manager.run_mla(q_nope, torch.zeros(4, NUM_HEADS, KPE), "main", layer)
        # reference over the ckv part only
        q = torch.cat([q_nope, torch.zeros(4, NUM_HEADS, KPE)], -1)
        torch.testing.assert_close(out, _reference(q, rows, 0), atol=1e-5, rtol=1e-5)


class TestBackendSelection:
    def _info(self, cfg: KVConfig) -> EngineResourceInfo:
        return EngineResourceInfo(
            device=torch.device("cpu"),
            dependencies={"kv": KVSpec(resource_key="kv", nodes={"llm"}, config=cfg)},
        )

    def test_mla_spec_builds_the_mla_manager(self):
        spec = AttentionSpec(
            resource_key="attn", nodes={"llm"},
            config=AttentionConfig(
                kv_cache="kv", backend=AttnBackend.MLA,
                mla_ckv_dim=CKV, softmax_scale=SCALE,
            ),
        )
        manager = AttentionManager.build(spec, self._info(_kv_config()))
        assert isinstance(manager, MLAAttentionManager)
        assert manager.ckv_dim == CKV and manager.kpe_dim == KPE
        assert manager.num_heads == NUM_HEADS

    def test_mla_spec_needs_its_dims_and_an_mla_cache(self):
        bad = AttentionSpec(
            resource_key="attn", nodes={"llm"},
            config=AttentionConfig(kv_cache="kv", backend=AttnBackend.MLA),
        )
        with pytest.raises(ValueError, match="mla_ckv_dim"):
            AttentionManager.build(bad, self._info(_kv_config()))
        nhd = KVConfig(
            num_layers=1, num_kv_heads=1, head_dim=HEAD_DIM, max_seq_len=8,
            max_num_pages=4, page_size=PAGE_SIZE, num_qo_heads=NUM_HEADS,
        )
        spec = AttentionSpec(
            resource_key="attn", nodes={"llm"},
            config=AttentionConfig(
                kv_cache="kv", backend=AttnBackend.MLA,
                mla_ckv_dim=CKV, softmax_scale=SCALE,
            ),
        )
        with pytest.raises(ValueError, match="KVLayout.MLA"):
            AttentionManager.build(spec, self._info(nhd))

    def test_yaml_override_can_name_the_backend(self):
        spec = AttentionSpec(
            resource_key="attn", nodes={"llm"},
            config=AttentionConfig(kv_cache="kv", mla_ckv_dim=CKV, softmax_scale=SCALE),
        )
        spec.apply_yaml_overrides(backend="mla")
        assert spec.config.backend is AttnBackend.MLA
