"""A layer set of another KV geometry (a draft model's heads and head size) in a paged pool: a storage of its own
over the pool's pages (`PagedKVConfig.join`, `KVStorage`, `KVCache.extra`).

A page id names one page in every storage, so reservation, take-back, eviction, offload and the prefix index all
treat a page as one unit. A pool with one storage is the pool it always was (test_drafter_layer_sets checks the
shipped toys bit for bit).
"""
from __future__ import annotations

import pytest
import torch

from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.cache import KVCache
from mstar.engine.resources.kv.config import (
    KVLayerSet,
    KVLayout,
    KVReqConfig,
    KVStep,
    KVStorage,
    PagedKVConfig,
    PrefixSpan,
    StreamLayout,
)
from mstar.engine.resources.kv.cpu_page_pool import CPUPagePool, PageCopies
from mstar.engine.resources.kv.keys import chain
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.step import Lag, Segment, StepContext
from mstar.model.submodule_base import ARNodeInputs

PAGE = 8
# the target's two layers of 2 KV heads of 8; a draft model of 4 KV heads of 64 (the RFC's toy geometry)
DRAFT = KVLayerSet("kv", "draft", 1, 4, 64)
MTP = KVLayerSet("kv", "mtp", 1, 2, 8)


class _StubTransfer:
    def __init__(self, transfer_engine_info, kv_cache, **kwargs):
        del transfer_engine_info, kv_cache, kwargs

    def get_kv_transfer_info(self, **kwargs):
        del kwargs

    def owns_transfer_info(self, transfer_info, **kwargs):
        del kwargs
        return transfer_info == self.get_kv_transfer_info()

    def remove_request(self, request_id):
        del request_id

    def start_async_retrieve(self, **kwargs):
        del kwargs

    def cleanup(self):
        pass


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)


def _config(*sets: KVLayerSet, **kw) -> PagedKVConfig:
    config = PagedKVConfig(num_layers=2, num_kv_heads=2, head_dim=8, max_seq_len=512, max_num_pages=32,
                           page_size=PAGE, **kw)
    for layer_set in sets:
        config.join(layer_set)
    return config


@pytest.mark.parametrize("order", [(DRAFT, MTP), (MTP, DRAFT)], ids=["other_first", "own_first"])
def test_an_extra_storage_follows_every_layer_of_the_pools_own_whatever_the_join_order(order):
    config = _config(*order)
    assert config.num_layers == 3 and config.total_layers == 4
    assert config.layer_sets == {"target": (0, 2), "mtp": (2, 1), "draft": (3, 1)}
    assert config.extra_storages == (KVStorage(1, 4, 64),)
    assert config.layer_set_storage("draft") == 1 and config.layer_set_storage("mtp") == 0
    cache = KVCache(config, torch.device("cpu"), torch.float32)
    assert list(cache.tensor.shape) == [3, 32, 2, PAGE, 2, 8]
    assert [list(t.shape) for t in cache.extra] == [[1, 32, 2, PAGE, 4, 64]]
    # global layer 3 is the draft model's, in its own storage
    assert cache.layer_view(3).data_ptr() == cache.extra[0][0].data_ptr()
    assert cache.layer_view(2).data_ptr() == cache.tensor[2].data_ptr()


def test_page_operations_and_sizes_cover_every_storage():
    config = _config(DRAFT)
    cache = KVCache(config, torch.device("cpu"), torch.float32)
    for t in cache.storages:
        t.copy_(torch.randn_like(t))
    before = [t.clone() for t in cache.storages]
    cache.copy_pages([3, 4], [7, 8])
    for t, old in zip(cache.storages, before, strict=True):
        assert torch.equal(t[:, 7], old[:, 3]) and torch.equal(t[:, 8], old[:, 4])
    cache.copy_slots(5, 9, 3)
    for t in cache.storages:
        assert torch.equal(t[:, 9, :, :3], t[:, 5, :, :3])
    assert cache.nbytes == cache.tensor.nbytes + cache.extra[0].nbytes
    assert cache.page_bytes == cache.tensor[:, 0].nbytes + cache.extra[0][:, 0].nbytes
    assert cache.layer_set_bytes_per_token() == {"target": 2 * 2 * 2 * 8 * 4, "draft": 1 * 2 * 4 * 64 * 4}
    # a write through the global layer lands in the draft model's storage
    k = torch.randn(2, 4, 64)
    cache.write_tokens(2, k, k + 1, torch.tensor([6, 6]), torch.tensor([0, 1]))
    assert torch.equal(cache.extra[0][0, 6, 0, :2], k)
    with pytest.raises(NotImplementedError, match="a layer set of its own KV geometry"):
        cache.chunk_ptrs(0, 1, 0, PAGE)


def test_an_unknown_set_and_query_heads_with_no_set_are_refused():
    from mstar.engine.resources.attn.config import AttentionConfig

    with pytest.raises(ValueError, match="no layer set 'nope'"):
        _config(DRAFT).geometry_of("nope")
    with pytest.raises(ValueError, match="needs layer_set"):
        AttentionConfig(kv_cache="kv", num_qo_heads=16)


def test_an_extra_storage_is_refused_where_it_cant_be_kept():
    with pytest.raises(ValueError, match="joined twice"):
        _config(DRAFT, DRAFT)
    mla = PagedKVConfig(num_layers=2, num_kv_heads=1, head_dim=8, max_seq_len=512, layout=KVLayout.MLA,
                        kv_lora_rank=6, qk_rope_head_dim=2)
    with pytest.raises(ValueError, match="MLA pool can.t store"):
        mla.join(DRAFT)
    windowed = _config()
    windowed.keep_window(32, (1,))
    with pytest.raises(ValueError, match="sliding window per page group"):
        windowed.join(DRAFT)
    joined = _config(DRAFT)
    with pytest.raises(ValueError, match="sliding window per page group"):
        joined.keep_window(32, (1,))
    with pytest.raises(ValueError, match="can't be read by another instance"):
        KVManager(cfg=_config(DRAFT), name="kv", joint_comm_group=None, transfer_engine_info=None,
                  device=torch.device("cpu"), dtype=torch.float32, needs_remote_transfer=True)


def _manager(*sets: KVLayerSet) -> KVManager:
    return KVManager(cfg=_config(*sets), name="kv", joint_comm_group=None, transfer_engine_info=None,
                     device=torch.device("cpu"), dtype=torch.float32)


def test_the_prefix_root_moves_only_with_an_extra_storage():
    plain, own, other = _manager(), _manager(MTP), _manager(DRAFT)
    assert plain.fingerprint() == own.fingerprint()
    assert other.fingerprint() != plain.fingerprint()


def _keys(n: int) -> list[bytes]:
    tokens = list(range(n))
    return chain([tokens[at:at + PAGE] for at in range(0, n, PAGE)])


def _step(kv: KVManager, rid: str, span: int) -> None:
    step = KVStep(segments=(Segment(rid, "main", span),))
    ctx = StepContext(request_ids=(rid,), graph_walk="prefill", slot=0, capture=False)
    assert kv.admit(step, ctx).ok
    kv.plan(step, ctx)
    kv.commit(step, ctx)


def test_a_prefix_hit_reuses_the_pages_of_every_storage():
    """The cold request's pages, held in both storages, are the warm request's: the draft model's KV at
    the cached positions is the cold request's, never recomputed."""
    kv = _manager(DRAFT)
    kv.enable_prefix_cache(b"a root")
    layout = {"main": StreamLayout(spans=(PrefixSpan(32, 32, "prefill"),), keyed_spans=1)}
    kv.ingest_request("cold", KVReqConfig(prefix_keys={"main": _keys(32)}, kv_layout=layout))
    _step(kv, "cold", 32)
    cold = list(kv._streams["cold"]["main"].page_indices)
    for t in kv.kv_cache.storages:
        t[:, cold] = torch.randn_like(t[:, cold])
    painted = [t[:, cold].clone() for t in kv.kv_cache.storages]
    kv.remove_request("cold")

    kv.ingest_request("warm", KVReqConfig(prefix_keys={"main": _keys(32)}, kv_layout=layout))
    inputs = ARNodeInputs(input_seq_len=32)
    hit = kv.resolve_cached_prefix("warm", "LLM", "prefill", inputs)
    assert hit is not None and hit.tokens == 3 * PAGE, "four cached pages make a hit of three"
    kv.apply_cached_prefix("warm", "LLM", "prefill", inputs, hit)
    _step(kv, "warm", 32 - hit.tokens)
    warm = list(kv._streams["warm"]["main"].page_indices)
    assert warm[:3] == cold[:3]
    for t, old in zip(kv.kv_cache.storages, painted, strict=True):
        assert torch.equal(t[:, warm[:3]], old[:, :3])
    kv.assert_pages_conserved()


def test_the_reload_check_fingerprints_every_storage():
    """A change confined to the draft model's storage changes the request's fingerprint
    (`KVManager.debug_fingerprint`, checked at reload under MSTAR_KV_DEBUG_ASSERTS)."""
    kv = _manager(DRAFT)
    kv.ingest_request("r0")
    _step(kv, "r0", 3 * PAGE)
    pages = list(kv._streams["r0"]["main"].page_indices)
    for t in kv.kv_cache.storages:
        t[:, pages] = torch.randn_like(t[:, pages])
    before = kv.debug_fingerprint("r0")
    kv.kv_cache.extra[0][0, pages[1], 0, 0, 0, 0] += 1.0
    after = kv.debug_fingerprint("r0")
    changed = [part for part in before if (not torch.equal(after[part], before[part]) if torch.is_tensor(before[part])
                                           else after[part] != before[part])]
    assert changed == ["kv/main/storage1"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="offload needs pinned host memory")
def test_an_offload_and_reload_bring_back_every_storage():
    kv = KVManager(cfg=_config(DRAFT, cpu_offload_pages=16), name="kv", joint_comm_group=None,
                   transfer_engine_info=None, device=torch.device("cuda"), dtype=torch.float32)
    assert [list(t.shape[2:]) for t in kv._cpu_pool.cpu_storages] == [list(t.shape[2:]) for t in kv.kv_cache.storages]
    kv.ingest_request("r0")
    _step(kv, "r0", 3 * PAGE)
    pages = list(kv._streams["r0"]["main"].page_indices)
    for t in kv.kv_cache.storages:
        t[:, pages] = torch.randn_like(t[:, pages])
    expected = [t[:, pages].clone() for t in kv.kv_cache.storages]
    assert kv.offload("r0") == 3
    for t in kv.kv_cache.storages:
        t.fill_(-7.0)
    assert kv.reload("r0") is True
    kv._cpu_pool.sync()
    torch.cuda.synchronize()
    pages = list(kv._streams["r0"]["main"].page_indices)
    for t, old in zip(kv.kv_cache.storages, expected, strict=True):
        assert torch.equal(t[:, pages], old)


class _RecordingCopier:
    """Records each page copy as (destination, its pages, source, its pages)."""

    def __init__(self):
        self.copies = []

    def copy(self, dst, dst_pages, src, src_pages, stream):
        del stream
        self.copies.append((id(dst), list(dst_pages), id(src), list(src_pages)))


class _Stream:
    def wait_stream(self, stream):
        del stream


def test_each_host_copy_covers_every_storage_primary_first(monkeypatch):
    """Offload, reload (and a prefix hit served from the host) and the prefix cache's host write each copy every
    storage of a pool with a drafter's own, with its host storage. A smallest real pool on the CPU: its own
    constructor pins host memory, which needs CUDA."""
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: None)
    kv_cache = _manager(DRAFT).kv_cache
    pool = CPUPagePool.__new__(CPUPagePool)
    pool.cpu_storages = [torch.empty_like(t) for t in kv_cache.storages]
    pool._copier = copier = _RecordingCopier()
    pool._stream = pool._write_stream = _Stream()
    pool._copied_since_write = False
    copies = PageCopies(gpu=[3, 5], cpu=[0, 1])

    pool.copy_out(kv_cache, copies)
    pool.copy_in(kv_cache, copies)
    pool.write(kv_cache, copies)

    pairs = [(id(host), id(device)) for host, device in zip(pool.cpu_storages, kv_cache.storages, strict=True)]
    assert len(pairs) == 2 and kv_cache.storages[0] is kv_cache.tensor
    to_host = [(host, [0, 1], device, [3, 5]) for host, device in pairs]
    to_device = [(device, [3, 5], host, [0, 1]) for host, device in pairs]
    assert copier.copies == to_host + to_device + to_host
    # a call site handing over the primary tensor alone would leave the drafter's pages behind
    with pytest.raises(ValueError):
        pool.copy_out(kv_cache.tensor, copies)


def _flashinfer_reason() -> str | None:
    if not torch.cuda.is_available():
        return "needs CUDA"
    try:
        import flashinfer  # noqa: F401
    except ImportError:
        return "needs flashinfer"
    return None


@pytest.mark.skipif(_flashinfer_reason() is not None, reason=str(_flashinfer_reason()))
def test_an_attention_binding_on_a_set_plans_and_reads_with_its_geometry():
    """A draft model's attention (16 query heads over 4 KV heads of 64) over its own storage of a target pool of
    2 KV heads of 128: FlashInfer planned with the set's geometry, run on the set's layer, against a dense read."""
    from mstar.engine.resources.attn.config import AttentionStep
    from mstar.engine.resources.attn.flashinfer import FlashInferManager
    from mstar.engine.resources.kv.plan import KVPlanOutput, KVPlanOutputs, PagedIndptrs, SequenceView

    page, ctx_len, rows_q = 16, 40, 4
    config = PagedKVConfig(num_layers=2, num_kv_heads=2, head_dim=128, num_qo_heads=4, max_seq_len=512,
                           max_num_pages=32, page_size=page)
    config.join(KVLayerSet("kv", "draft", 1, 4, 64))
    device = torch.device("cuda")
    cache = KVCache(config, device, torch.bfloat16)
    view = config.geometry_of("draft", 16)
    assert (view.num_kv_heads, view.head_dim, view.num_qo_heads, view.num_units) == (4, 64, 16, config.num_units)
    manager = FlashInferManager(kv_cache="kv", device=device, dtype=torch.bfloat16, kv_config=view)
    pages = [1, 2, 3]
    layer = cache.layer_view(2)
    assert layer.data_ptr() == cache.extra[0][0].data_ptr()
    gen = torch.Generator(device="cuda").manual_seed(0)
    layer[pages] = torch.randn(layer[pages].shape, generator=gen, device=device, dtype=torch.bfloat16)
    views = [SequenceView(request_id="r0", label="main", page_idxs=pages, length=ctx_len, to_compute=rows_q)]
    indptrs = PagedIndptrs(
        qo_indptr=torch.tensor([0, rows_q], dtype=torch.int32),
        paged_kv_indptr=torch.tensor([0, len(pages)], dtype=torch.int32),
        paged_kv_indices=torch.tensor(pages, dtype=torch.int32),
        paged_kv_last_page_len=torch.tensor([ctx_len - (len(pages) - 1) * page], dtype=torch.int32),
    )
    step_ctx = StepContext(request_ids=("r0",), graph_walk="prefill", slot=0, capture=False)
    step_ctx.plan_results = {"kv": KVPlanOutputs({"main": KVPlanOutput(cpu_indptrs=indptrs, views=views)})}
    manager.plan(AttentionStep(causal=True), step_ctx)
    q = torch.randn(rows_q, 16, 64, generator=gen, device=device, dtype=torch.bfloat16)
    out = manager.run(q, "main", kv_cache_layer=layer)

    pos = torch.arange(ctx_len, device=device)
    kv = layer[torch.tensor(pages, device=device)[pos // page], :, pos % page].float()  # [ctx, 2, 4, 64]
    k, v = kv[:, 0].repeat_interleave(4, 1), kv[:, 1].repeat_interleave(4, 1)
    at = torch.arange(ctx_len - rows_q, ctx_len, device=device)
    mask = pos[None, :] <= at[:, None]
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.float().transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), attn_mask=mask,
    ).transpose(0, 1)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


def test_startup_takes_a_drafter_of_its_own_geometry_only_with_a_binding_on_its_set():
    from mstar.engine.resources.attn.config import AttentionConfig, AttentionSpec
    from mstar.engine.resources.draft.config import DraftConfig
    from mstar.engine.resources.draft.drafter import ChunkBoundary, DrafterNeeds, ModelGeometry, VerifyNode
    from mstar.engine.resources.draft.startup import geometry_problems
    from mstar.engine.resources.kv.config import KVSpec

    drafter = ModelGeometry(num_layers=1, num_kv_heads=4, head_dim=64, hidden_size=32, vocab_size=97,
                            dtype="bfloat16", num_qo_heads=16)
    target = ModelGeometry(num_layers=2, num_kv_heads=2, head_dim=128, hidden_size=32, vocab_size=97,
                           dtype="bfloat16")
    needs = DrafterNeeds(kv=drafter.layer_set("kv", "draft"), chunk_boundary=ChunkBoundary.NONE, geometry=drafter)
    decl = VerifyNode("LLM", "decode", DraftConfig("dspark", 3, model="d"), needs, target=target)
    pool = KVSpec(resource_key="kv", nodes={"LLM"}, config=PagedKVConfig(
        num_layers=2, num_kv_heads=2, head_dim=128, num_qo_heads=4, max_seq_len=512))
    pool.config.join(needs.kv)
    target_attn = AttentionSpec(resource_key="attn", nodes={"LLM"}, config=AttentionConfig(kv_cache="kv"))
    specs = {"kv": pool, "attn": target_attn}
    assert any("no attention over KV 'kv' plans with it" in p for p in geometry_problems(decl, specs))

    def bound(qo):
        return AttentionSpec(resource_key="draft_attn", nodes={"LLM"},
                             config=AttentionConfig(kv_cache="kv", layer_set="draft", num_qo_heads=qo))

    assert geometry_problems(decl, {**specs, "draft_attn": bound(16)}) == []
    assert any("has 16 query heads; the attention over KV 'kv' plans 8" in p
               for p in geometry_problems(decl, {**specs, "draft_attn": bound(8)}))


@pytest.mark.parametrize("shards, target_heads, draft_heads", [(1, 2, 4), (2, 1, 2), (4, 1, 1), (8, 1, 1)])
def test_each_storage_shards_its_own_heads_replicating_past_its_count(shards, target_heads, draft_heads):
    """A rank holds a slice of each storage's KV heads, or one replicated head when a storage has fewer
    heads than ranks; the attention binding's view of a set shards by the same rule."""
    config = PagedKVConfig(num_layers=2, num_kv_heads=2, head_dim=8, num_qo_heads=8, max_seq_len=512,
                           max_num_pages=32, page_size=PAGE)
    config.join(DRAFT)
    config.shard(shards)
    assert config.num_kv_heads == target_heads and config.storage_kv_heads(1) == draft_heads
    cache = KVCache(config, torch.device("cpu"), torch.float32)
    assert cache.tensor.shape[-2] == target_heads and cache.extra[0].shape[-2] == draft_heads
    view = config.geometry_of("draft", 16)
    view.shard(shards)
    assert (view.num_kv_heads, view.num_qo_heads) == (draft_heads, 16 // shards)


def test_a_layer_of_an_extra_storage_writes_and_reads_through_the_manager():
    """The pool's write and read paths (`write_kv` / `read_kv`) reach an extra storage's layers too."""
    kv = _manager(DRAFT)
    kv.ingest_request("r0")
    _step(kv, "r0", 2 * PAGE)
    assert kv._layer_group == [0, 0, 0]
    pages = list(kv._streams["r0"]["main"].page_indices)
    k = torch.randn(3, 4, 64)
    kv.kv_cache.write_tokens(2, k, k + 1, torch.tensor([pages[0]] * 3), torch.tensor([0, 1, 2]))
    assert torch.equal(kv.kv_cache.read_tokens(2, torch.tensor([pages[0]] * 3), torch.tensor([0, 1, 2]))[:, 0], k)


def _plan(kv: KVManager, rid: str, span: int, lag: dict | None = None) -> KVStep:
    step = KVStep(segments=(Segment(rid, "main", span),), lag=lag or {})
    ctx = StepContext(request_ids=(rid,), graph_walk="verify", slot=0, capture=False)
    assert kv.admit(step, ctx).ok
    kv.plan(step, ctx)
    return step, ctx


def _written(kv: KVManager, rid: str, layer: int, p: int | None = None) -> list[int | None]:
    """The stream positions layer ``layer`` writes this step, in pass ``p`` (None: the sink page)."""
    kv.select_pass(p)
    state = kv._write_state("main", layer)
    n = state.total_tokens
    pages = list(kv._streams[rid]["main"].page_indices)
    return [None if page == manager_mod.SINK_PAGE else pages.index(page) * PAGE + offset
            for page, offset in zip(state.token_to_page[:n].tolist(), state.token_to_cache[:n].tolist(), strict=True)]


LAG = Lag(1, (2, 1, 1))


@pytest.mark.parametrize("drafter", [MTP, DRAFT], ids=["own_storage", "extra_storage"])
def test_a_lagged_set_writes_each_token_one_position_back_one_pass_at_a_time(drafter):
    """`KVStep.lag`: a draft model's layers write a verify block one position behind the target's, in its
    passes (``[prev, bonus]``, then a row each), row 0 (the token before the block) to the sink unless the
    step keeps it; the next unlagged step is unlagged."""
    kv = _manager(drafter)
    kv.ingest_request("r0")
    _step(kv, "r0", 2 * PAGE)  # L = 16: the lagged row 0 is the last slot of the previous page
    draft = kv.layer_set(drafter.name).first
    step, ctx = _plan(kv, "r0", 4, lag={drafter.name: LAG})
    assert _written(kv, "r0", 0) == [16, 17, 18, 19]
    assert [_written(kv, "r0", draft, p) for p in range(3)] == [[15, 16], [17], [18]]
    kv.sink_lagged_rows(torch.tensor([False]))
    assert [_written(kv, "r0", draft, p) for p in range(3)] == [[None, 16], [17], [18]]
    with pytest.raises(RuntimeError, match="outside a pass"):
        _written(kv, "r0", draft)

    # the write itself lands in the set's storage at the pass's lagged slots
    k = torch.randn(1, drafter.num_kv_heads, drafter.head_dim)
    kv.select_pass(1)
    kv.write_kv(k, k + 1, layer_idx=draft, label="main")
    pages = list(kv._streams["r0"]["main"].page_indices)
    assert torch.equal(kv.kv_cache.read_tokens(draft, torch.tensor([pages[2]]), torch.tensor([1]))[:, 0], k)
    kv.select_pass(None)
    kv.commit(step, ctx)

    _plan(kv, "r0", 4, lag={drafter.name: LAG})
    kv.sink_lagged_rows(torch.tensor([True]))
    assert [_written(kv, "r0", draft, p) for p in range(3)] == [[19, 20], [21], [22]]
    # an unlagged step after a lagged one: no lagged addressing is left behind
    _plan(kv, "r0", 2)
    assert _written(kv, "r0", draft) == _written(kv, "r0", 0) == [20, 21]


def test_a_lag_names_a_set_of_the_pool_and_its_passes_cover_the_block():
    kv = _manager(MTP)
    kv.ingest_request("r0")
    _step(kv, "r0", PAGE)
    with pytest.raises(KeyError, match="doesn't hold"):
        _plan(kv, "r0", 4, lag={"nope": LAG})
    with pytest.raises(ValueError, match="cover 4 rows"):
        _plan(kv, "r0", 4, lag={"mtp": Lag(1, (2, 1))})


def test_a_lagged_row_0_never_rewrites_a_sealed_prompt_page():
    """A prompt of whole pages ends on a keyed (sealed, published) page, whose last slot is a request's first
    lagged row 0 (L - 1): the step's keep flag is its default 0 there (the prompt step wrote that token), so
    the row goes to the sink and the page's bytes stay as the prompt step left them."""
    kv = _manager(MTP)
    kv.enable_prefix_cache(b"a root")
    layout = {"main": StreamLayout(spans=(PrefixSpan(2 * PAGE, 2 * PAGE, "prefill"),), keyed_spans=1)}
    kv.ingest_request("r0", KVReqConfig(prefix_keys={"main": _keys(2 * PAGE)}, kv_layout=layout))
    _step(kv, "r0", 2 * PAGE)
    draft = kv.layer_set("mtp").first
    pages = list(kv._streams["r0"]["main"].page_indices)
    kv.kv_cache.tensor[:, pages[1]] = torch.randn_like(kv.kv_cache.tensor[:, pages[1]])
    sealed = kv.kv_cache.tensor[:, pages[1]].clone()
    _plan(kv, "r0", 4, lag={"mtp": LAG})
    kv.sink_lagged_rows(torch.tensor([False]))
    assert _written(kv, "r0", draft, 0) == [None, 2 * PAGE]
    k = torch.randn(2, MTP.num_kv_heads, MTP.head_dim)
    kv.select_pass(0)
    kv.write_kv(k, k + 1, layer_idx=draft, label="main")
    kv.select_pass(None)
    assert torch.equal(kv.kv_cache.tensor[:, pages[1]], sealed)
