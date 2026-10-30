"""dsa_long_context: index keys on the MLA cache's pages, slot selection, sparse attention (CPU
paths).

Decode past index_topk is held, layer by layer, to the reference indexer's selection and a
masked dense softmax. Prefill: every row selects over its own causal prefix, so a prompt
prefilled past index_topk equals prefilling index_topk tokens and decoding the rest one at a
time.
"""
import sys
import types

import pytest
import torch


def _cpu_rmsnorm(x, weight, eps=1e-6):
    x32 = x.float()
    return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps) * weight.float()).to(x.dtype)


def _cpu_flashinfer() -> types.ModuleType:
    fi = types.ModuleType("flashinfer")
    fi.norm = types.SimpleNamespace(rmsnorm=_cpu_rmsnorm)
    return fi


from mstar.engine.resources import StepContext  # noqa: E402
from mstar.engine.resources.dsa import paged, sparse_mla  # noqa: E402
from mstar.engine.resources.dsa.config import INDEX_PLANES  # noqa: E402
from mstar.engine.resources.dsa.paged import DSAContext  # noqa: E402
from mstar.model.glm52._testing import (  # noqa: E402
    build_cpu_resources,
    build_random_model,
    index_topk,
    install_cpu_rope,
)
from mstar.model.glm52.components.attention import Glm52MLAAttention  # noqa: E402
from mstar.model.glm52.components.indexer import full_indexer_layers  # noqa: E402
from mstar.model.glm52.config import (  # noqa: E402
    ATTN_RESOURCE,
    DSA_RESOURCE,
    KV_RESOURCE,
    Glm52ModelConfig,
)
from mstar.model.glm52.submodules import Glm52LLMSubmodule  # noqa: E402
from mstar.model.submodule_base import ARNodeInputs, ModelInputsFromEngine  # noqa: E402


@pytest.fixture(autouse=True)
def _force_cpu_flashinfer(monkeypatch):
    """CPU tensors throughout: the stub, whatever flashinfer is installed."""
    monkeypatch.setitem(sys.modules, "flashinfer", _cpu_flashinfer())
    install_cpu_rope(monkeypatch)


def _cfg(topk):
    cfg = Glm52ModelConfig.reduced()
    cfg.mla_absorb = True
    cfg.dsa_long_context = True
    cfg.index_topk = topk
    return cfg


class _Driver:
    """declare -> admit -> plan -> preprocess -> forward -> commit, one request."""

    def __init__(self, sub, cfg, rid="r0", page_size=4):
        self.sub, self.rid = sub, rid
        self.resources, self.runner = build_cpu_resources(cfg, [rid], page_size=page_size)
        sub.bind_node_resources(self.resources)

    def step(self, walk, ids):
        ar = ARNodeInputs(input_ids=ids, input_seq_len=ids.shape[0])
        step = self.sub.declare_step(walk, [self.rid], [ar])
        step.set_ctx(StepContext(request_ids=(self.rid,), graph_walk=walk, slot=0, capture=False))
        assert self.runner.admit(step).ok
        self.runner.plan(step)
        engine_inputs = ModelInputsFromEngine(
            request_ids=[self.rid], per_request_info={}, resources=self.resources, step=step)
        packed = self.sub.preprocess(walk, engine_inputs, [ar])
        self.last_ctx = self.sub._dsa_ctx(packed)
        with torch.no_grad():
            logits = self.sub.forward(walk, engine_inputs, **packed)["logits"]
        self.runner.commit(step)
        return logits


def _serve(sub, cfg, prompt, decode):
    driver = _Driver(sub, cfg)
    out = [driver.step("prefill", prompt)[0]]
    out += [driver.step("decode", t.view(1))[0] for t in decode]
    return out, driver


def test_paged_resources_and_step():
    cfg = _cfg(topk=4)
    from mstar.model.glm52.glm52_model import Glm52Model

    model = object.__new__(Glm52Model)
    model.config = cfg
    specs = {s.resource_key: s for s in model.get_node_resources()}
    # the index keys sit on the latent cache's pages: a row per token per FULL layer
    index = specs[KV_RESOURCE].config.aux_planes[INDEX_PLANES]
    assert index.num_layers == len(full_indexer_layers(cfg)) >= 1
    assert index.width == cfg.index_head_dim
    # the DSA resource selects over them and attends over the latents
    dsa = specs[DSA_RESOURCE].config
    assert specs[DSA_RESOURCE].depends_on() == {KV_RESOURCE}
    assert (dsa.topk, dsa.kpool, dsa.dense_below_topk) == (cfg.index_topk, 1, True)
    sub = Glm52LLMSubmodule(language_model=build_random_model(cfg, seed=0), config=cfg)
    ar = ARNodeInputs(input_ids=torch.zeros(3, dtype=torch.long), input_seq_len=3)
    steps = sub.declare_step("prefill", ["r0"], [ar]).steps
    assert KV_RESOURCE in steps and DSA_RESOURCE in steps


def _positions(slots, table, page_size):
    """Request positions of latent-cache slots, by the request's pages."""
    where = {int(p): j for j, p in enumerate(table.tolist())}
    return [where[int(s) // page_size] * page_size + int(s) % page_size for s in slots]


def _masked_reference(cache, table, page_size, n_keys, picked, query, scale, rank):
    """Dense MQA over a request's first ``n_keys`` cached latents with every position outside
    ``picked`` masked; shares no code with dsa/paged.py or sparse_mla."""
    pos = torch.arange(n_keys)
    flat = cache[table[pos // page_size].long(), pos % page_size].float()
    mask = torch.full((n_keys,), float("-inf"))
    mask[picked] = 0.0
    weights = ((query.float() @ flat.T) * scale + mask).softmax(-1)
    return weights @ flat[:, :rank]


def test_paged_decode_across_topk_matches_an_independent_reference(monkeypatch):
    """Every decode step past index_topk: a FULL layer's latents are the reference indexer's
    top index_topk positions over the stored keys (its queries scored by index_topk), and every
    layer's attention is a dense softmax over the cache masked to them."""
    cfg = _cfg(topk=4)
    sub = Glm52LLMSubmodule(language_model=build_random_model(cfg, seed=0), config=cfg)
    update, attend = Glm52MLAAttention._dsa_update, Glm52MLAAttention._attend_selected
    selected, attended = [], []

    def checked_update(self, hidden, q_c, position_ids):
        selection = update(self, hidden, q_c, position_ids)
        if self.indexer is not None and selection is not None:
            ctx = self._dsa.ctx  # the forward's, open on the DSA resource
            n, page = ctx.host_pos[0] + 1, ctx.page_size
            pos = torch.arange(n)
            store = self._kv.aux_view(INDEX_PLANES, self.index_layer_idx)
            keys = store[ctx.pages[pos // page].long(), pos % page]
            q, w = self.indexer.compute_query(q_c, hidden, position_ids)
            want = index_topk(q, w, keys, position_ids, ctx.topk)[0]
            got = _positions(selection[0, :ctx.topk], ctx.pages, page)
            assert sorted(got) == sorted(want.tolist()), (self.layer_idx, n)
            selected.append(n)
        return selection

    def checked_attend(self, slots, q_nope, q_pe, latent, dsa_selection=None):
        ctx = self._dsa.ctx
        out = attend(self, slots, q_nope, q_pe, latent)
        n, page = ctx.host_pos[0] + 1, ctx.page_size
        picked = _positions(slots[0, :ctx.topk], ctx.pages, page)
        assert len(picked) == ctx.topk < n  # a strict subset of the keys the row sees
        cache = self._kv.layer_view(self.cache_layer_idx)
        query = torch.cat([q_nope, q_pe], dim=-1)[0]
        args = (cache, ctx.pages, page, n)
        ref = _masked_reference(*args, picked, query, self.softmax_scale, q_nope.shape[-1])
        torch.testing.assert_close(out[0].float(), ref, rtol=1e-5, atol=1e-5)
        # discrimination: attending every key the row sees must not match
        dense = _masked_reference(*args, list(range(n)), query, self.softmax_scale,
                                  q_nope.shape[-1])
        assert not torch.allclose(out[0].float(), dense, rtol=1e-3, atol=1e-3)
        attended.append((self.layer_idx, n))
        return out

    monkeypatch.setattr(Glm52MLAAttention, "_dsa_update", checked_update)
    monkeypatch.setattr(Glm52MLAAttention, "_attend_selected", checked_attend)
    decode = torch.tensor([7, 1, 8, 3, 6, 4])
    _, driver = _serve(sub, cfg, torch.tensor([5, 9, 2]), decode)
    assert isinstance(driver.last_ctx, DSAContext)
    assert driver.resources[KV_RESOURCE].stored_len("r0") == 3 + len(decode)
    # contexts 5..9 select: the FULL layer picks, the FULL and the SHARED layer attend
    assert selected == [5, 6, 7, 8, 9]
    assert attended == [(layer, n) for n in range(5, 10) for layer in (0, 1)]


def test_paged_prefill_beyond_topk_equals_prefill_then_decode():
    cfg = _cfg(topk=4)
    sub = Glm52LLMSubmodule(language_model=build_random_model(cfg, seed=1), config=cfg)
    tokens = torch.tensor([5, 9, 2, 7, 1, 8, 3, 6, 4, 11])  # 10 > topk: every row past 4 selects
    whole = _Driver(sub, cfg)
    logits_whole = whole.step("prefill", tokens)
    assert whole.last_ctx.needs_selection
    stepwise, _ = _serve(sub, cfg, tokens[:4], tokens[4:])
    torch.testing.assert_close(logits_whole[-1], stepwise[-1], rtol=1e-4, atol=1e-5)


def test_prefill_beyond_topk_selects():
    cfg = _cfg(topk=4)
    sub = Glm52LLMSubmodule(language_model=build_random_model(cfg, seed=0), config=cfg)
    driver = _Driver(sub, cfg)
    assert torch.isfinite(driver.step("prefill", torch.arange(1, 11))[0]).all()
    assert driver.last_ctx.needs_selection


@pytest.mark.parametrize("budget_rows", [1, 3])
def test_prefill_chunks_join_at_their_boundaries(monkeypatch, budget_rows):
    """Prefill rows are scored in chunks under a byte budget for their fp32 scores; chunks of
    a few rows, down to one, give the same logits."""
    cfg = _cfg(topk=4)
    sub = Glm52LLMSubmodule(language_model=build_random_model(cfg, seed=2), config=cfg)
    tokens = torch.tensor([5, 9, 2, 7, 1, 8, 3, 6, 4, 11, 13])
    whole = _Driver(sub, cfg).step("prefill", tokens)
    # rows of 11 keys' scores each
    monkeypatch.setattr(paged, "PREFILL_SCORE_BYTES", 4 * 11 * budget_rows)
    chunked = _Driver(sub, cfg).step("prefill", tokens)
    torch.testing.assert_close(chunked[-1], whole[-1], rtol=0, atol=0)


def test_long_prefill_runs_in_row_chunks():
    """A prefill longer than prefill_chunk_tokens runs the trunk over row chunks, each writing
    its keys and latents before the next selects over them: the one-pass logits."""
    tokens = torch.tensor([5, 9, 2, 7, 1, 8, 3, 6, 4, 11, 13])
    cfg = _cfg(topk=4)
    sub = Glm52LLMSubmodule(language_model=build_random_model(cfg, seed=3), config=cfg)
    whole = _Driver(sub, cfg).step("prefill", tokens)
    cfg.prefill_chunk_tokens = 3
    passes = []
    hidden = sub._hidden
    sub._hidden = lambda ids, *a, **k: passes.append(ids.shape[0]) or hidden(ids, *a, **k)
    chunked = _Driver(sub, cfg).step("prefill", tokens)
    assert passes == [3, 3, 3, 2]
    torch.testing.assert_close(chunked[-1], whole[-1], rtol=1e-5, atol=1e-6)
    # row chunks attend sparse on every row: the step plans no dense attention
    ar = ARNodeInputs(input_ids=tokens, input_seq_len=tokens.shape[0])
    assert ATTN_RESOURCE not in sub.declare_step("prefill", ["r0"], [ar]).steps
    short = ARNodeInputs(input_ids=tokens[:3], input_seq_len=3)
    assert ATTN_RESOURCE in sub.declare_step("prefill", ["r0"], [short]).steps


def test_batched_long_prefill_chunks_cut_across_requests(monkeypatch):
    """Row chunks of a packed prefill cut through both requests; each request's last row
    still gets its one-pass logits."""
    from mstar.model.glm52._testing import build_cpu_resources

    cfg = _cfg(topk=4)
    sub = Glm52LLMSubmodule(language_model=build_random_model(cfg, seed=4), config=cfg)
    prompts = [torch.tensor([5, 9, 2, 7, 1, 8, 3]), torch.tensor([6, 4, 11, 13, 2, 9, 5, 1, 7])]
    rids = ["a", "b"]

    def run(chunk):
        cfg.prefill_chunk_tokens = chunk
        resources, runner = build_cpu_resources(cfg, rids, page_size=4)
        sub.bind_node_resources(resources)
        ars = [ARNodeInputs(input_ids=p, input_seq_len=p.shape[0]) for p in prompts]
        step = sub.declare_step("prefill", rids, ars)
        step.set_ctx(StepContext(request_ids=tuple(rids), graph_walk="prefill", slot=0,
                                 capture=False))
        assert runner.admit(step).ok
        runner.plan(step)
        engine_inputs = ModelInputsFromEngine(
            request_ids=rids, per_request_info={}, resources=resources, step=step)
        packed = sub.preprocess("prefill", engine_inputs, ars)
        out = {}
        monkeypatch.setattr(sub, "_sample", lambda _, logits: out.setdefault("logits", logits))
        with torch.no_grad():
            sub.forward_batched("prefill", engine_inputs, **packed)
        runner.commit(step)
        return out["logits"]

    whole = run(8192)
    assert whole.shape[0] == 2
    torch.testing.assert_close(run(4), whole, rtol=1e-5, atol=1e-6)


def test_second_prefill_step_selects_over_the_stored_keys():
    """A prompt prefilled over two steps: the second step's rows sit at absolute positions and
    select over the keys the first step stored, giving the one-step logits."""
    tokens = torch.tensor([5, 9, 2, 7, 1, 8, 3, 6, 4, 11, 13])
    cfg = _cfg(topk=4)
    sub = Glm52LLMSubmodule(language_model=build_random_model(cfg, seed=5), config=cfg)
    whole = _Driver(sub, cfg).step("prefill", tokens)
    driver = _Driver(sub, cfg)
    driver.step("prefill", tokens[:6])
    second = driver.step("prefill", tokens[6:])
    assert driver.last_ctx.host_pos == list(range(6, 11))
    assert driver.resources[KV_RESOURCE].stored_len("r0") == len(tokens)
    torch.testing.assert_close(second[-1], whole[-1], rtol=1e-5, atol=1e-6)


def test_index_planes_hold_the_mtp_layer():
    """With MTP the draft layer's index keys take the plane after the trunk's."""
    from mstar.model.glm52.components.indexer import index_store_layer
    from mstar.model.glm52.glm52_model import Glm52Model

    cfg = _cfg(topk=4)
    cfg.num_hidden_layers, cfg.mtp_num_draft_tokens = 4, 2
    model = object.__new__(Glm52Model)
    model.config = cfg
    specs = {s.resource_key: s for s in model.get_node_resources()}
    trunk = len(full_indexer_layers(cfg))
    assert specs[KV_RESOURCE].config.aux_planes[INDEX_PLANES].num_layers == trunk + 1
    assert index_store_layer(cfg, cfg.num_hidden_layers) == trunk


class _FakeGroup:
    """A TP group of ``world`` ranks run one after another: each select call is one rank's, and
    all_gather returns every rank's block so far (a rank not yet run gives -7s)."""

    def __init__(self, world):
        self.world_size, self.rank, self.blocks = world, 0, {}

    def all_gather(self, t, dim=0):
        self.blocks[self.rank] = t
        filler = torch.full_like(t, -7)
        return torch.cat([self.blocks.get(r, filler) for r in range(self.world_size)], dim=dim)


def test_sharded_prefill_selection_equals_unsharded(monkeypatch):
    """Each rank's block of a long span, gathered rank-major, is the unsharded selection; a
    span below the shard threshold and a decode row are selected whole on every rank."""
    monkeypatch.setattr(sparse_mla, "SHARD_MIN_ROWS", 4)
    torch.manual_seed(9)
    nh, d, page, topk = 4, 8, 4, 16
    spans_n = [61, 9, 1]  # rows per request: sharded, too short to shard, decode
    ctx_lens = [61, 30, 40]  # each request's context after this step
    pages = [-(-n // page) for n in ctx_lens]
    index_layer = torch.randn(sum(pages), page, d)
    starts = torch.tensor([0, *pages]).cumsum(0)[:-1].tolist()
    row_req, positions, spans = [], [], []
    for i, (n, c) in enumerate(zip(spans_n, ctx_lens, strict=True)):
        spans.append((len(row_req), n, i))
        row_req += [i] * n
        positions += list(range(c - n, c))
    ctx = DSAContext(
        pos=torch.tensor(positions, dtype=torch.int32),
        row_req=torch.tensor(row_req, dtype=torch.int32),
        pages=torch.arange(sum(pages), dtype=torch.int32),
        page_start=torch.tensor(starts, dtype=torch.int32), host_pos=positions, spans=spans,
        host_page_start=starts, max_pools=max(ctx_lens), page_size=page, topk=topk, kpool=1)
    q, w = torch.randn(len(positions), nh, d), torch.randn(len(positions), nh)
    want = paged.select(q, w, index_layer, ctx)
    group = _FakeGroup(4)
    for rank in range(4):  # the last rank's call sees every block
        group.rank = rank
        got = paged.select(q, w, index_layer, ctx, group)
    assert torch.equal(got, want)


def test_a_prefill_bucket_pads_its_rows_and_scores_over_them():
    """A captured prefill bucket's rows: the real ones, then padding staged as position 0 of
    the first request; every row scores on its own over the bucket's width, and a bucket no
    wider than index_topk selects nothing."""
    from mstar.engine.resources.step import BucketKey, SlotLease

    cfg = _cfg(topk=8)
    sub = Glm52LLMSubmodule(language_model=build_random_model(cfg, seed=5), config=cfg)
    rids = ["a", "b"]
    ars = [ARNodeInputs(input_ids=p, input_seq_len=p.shape[0])
           for p in (torch.tensor([5, 9, 2]), torch.tensor([6, 4, 11, 13]))]
    resources, runner = build_cpu_resources(cfg, rids, page_size=4)
    sub.bind_node_resources(resources)
    step = sub.declare_step("prefill", rids, ars)
    ctx = StepContext(request_ids=tuple(rids), graph_walk="prefill", slot=0, capture=False)
    step.set_ctx(ctx)
    assert runner.admit(step).ok
    runner.plan(step)
    engine_inputs = ModelInputsFromEngine(
        request_ids=rids, per_request_info={}, resources=resources, step=step, captured=True)
    for rows in (8, 16):
        ctx.slot_lease = SlotLease(slot=0, bucket=BucketKey("prefill", 2, rows))
        dsa = sub._dsa_ctx(sub.preprocess("prefill", engine_inputs, ars))
        pad = rows - 7
        assert dsa.pos.tolist() == [0, 1, 2, 0, 1, 2, 3] + [0] * pad
        assert dsa.row_req.tolist() == [0, 0, 0, 1, 1, 1, 1] + [0] * pad
        assert dsa.host_pos is None and dsa.spans is None  # a replay's host values are stale
        assert dsa.max_pools == rows and dsa.decode_rows
        assert dsa.needs_selection == (rows > cfg.index_topk)
    # rows past a stored prefix see more keys than a bucket of their size scores: refused
    driver = _Driver(sub, cfg)
    driver.step("prefill", torch.tensor([5, 9, 2, 7, 1, 8]))
    ar = ARNodeInputs(input_ids=torch.tensor([3, 6, 4]), input_seq_len=3)
    step = sub.declare_step("prefill", [driver.rid], [ar])
    ctx = StepContext(request_ids=(driver.rid,), graph_walk="prefill", slot=0, capture=False)
    step.set_ctx(ctx)
    assert driver.runner.admit(step).ok
    driver.runner.plan(step)
    ctx.slot_lease = SlotLease(slot=0, bucket=BucketKey("prefill", 1, 4))
    engine_inputs = ModelInputsFromEngine(
        request_ids=[driver.rid], per_request_info={}, resources=driver.resources, step=step,
        captured=True)
    with pytest.raises(RuntimeError, match="sits at 8, past its 4-token bucket"):
        sub.preprocess("prefill", engine_inputs, [ar])


def test_a_prefill_past_a_cached_prefix_runs_eager():
    """A request whose prefill starts past a cached prefix sees more keys than its bucket's
    capture scores: its step declares a capture key no bucket has, until its prefill ran."""
    from mstar.conductor.request_info import CurrentForwardPassInfo

    cfg = _cfg(topk=8)
    sub = Glm52LLMSubmodule(language_model=build_random_model(cfg, seed=5), config=cfg)
    info = CurrentForwardPassInfo(request_id="a", graph_walk="prefill", fwd_index=0,
                                  random_seed=0, max_tokens=4, rid_handle=7)
    ar = ARNodeInputs(input_ids=torch.arange(10), input_seq_len=10)
    assert sub.cg_key_info("prefill", {7: info}) is None
    # the engine's call (reserve_replay_slot)
    assert sub.cg_key_info("prefill", {7: info}, per_request_input_metadata={}) is None
    cut = sub.split_inputs("prefill", info, ar, 6, 10)
    assert cut.input_seq_len == 4
    assert sub.cg_key_info("prefill", {7: info, 8: info}) == "eager"
    assert sub.declare_step("prefill", [7], [cut]).cg_key_info == "eager"
    assert sub.cg_key_info("decode", {7: info}) is None
    sub.cleanup_request(7)
    assert sub.cg_key_info("prefill", {7: info}) is None


def test_a_captured_prefill_never_runs_in_chunks():
    """A bucket wider than prefill_chunk_tokens still runs as one pass when captured: the
    chunked path plans eagerly, which a capture cannot."""
    cfg = _cfg(topk=8)
    cfg.prefill_chunk_tokens = 4
    sub = Glm52LLMSubmodule(language_model=build_random_model(cfg, seed=5), config=cfg)
    ids = torch.arange(8)
    ctx = types.SimpleNamespace(decode_rows=True)
    assert not sub._long_prefill("prefill", ids, ctx)
    ctx.decode_rows = False
    assert sub._long_prefill("prefill", ids, ctx)
