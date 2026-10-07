"""glm5_next DSA past index_topk (dsa.py) on CPU: the k-pool selection and pool keys against the
reference indexer (transformers' Glm5NextTextIndexer, ported below), and the long-context engine
path against itself (one-shot prefill vs stepwise decode vs windowed prefill) and against dense
MLA where they must agree.
"""
from __future__ import annotations

import dataclasses
import sys
import types

sys.path.insert(0, ".")

import pytest
import torch

from mstar.communication.tensors import LocalTransferEngine
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.communication import WorkerParallelGroups
from mstar.engine.engine import Engine, ExecutingBatch
from mstar.engine.resources import SamplingReqConfig
from mstar.engine.resources.base import Resource
from mstar.engine.resources.kv import manager as kv_mod
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.engine.resources.step import StepContext
from mstar.model.glm5_next import dsa
from mstar.model.glm5_next.components.causal_lm import Glm5NextForCausalLM
from mstar.model.glm5_next.config import KV_CACHE, SAMPLER, Glm5NextModelConfig
from mstar.model.glm5_next.glm5_next_model import Glm5NextModel, process_weights_after_loading
from mstar.model.glm5_next.submodules import Glm5NextLLMSubmodule
from mstar.model.submodule_base import InputMetadata

# -- the reference indexer (transformers modeling_glm5_next.Glm5NextTextIndexer), unpadded ----


def _ref_pools(k, gate, ape, kpool):
    """get_pooled_states for one unpadded sequence: pool keys (P, D), member indices (P, KP)."""
    s = k.shape[0]
    n = (s + kpool - 1) // kpool
    idx = torch.arange(n * kpool).view(n, kpool)
    valid = idx < s
    safe = idx.clamp(0, s - 1)
    logits = gate[safe].float() + ape.float()[None]
    logits = logits.masked_fill(~valid[..., None], float("-inf"))
    prob = torch.nan_to_num(logits.softmax(dim=1)).to(k.dtype)
    keys = (prob * k[safe]).sum(dim=1)
    pool_valid = valid.all(-1)
    return keys[pool_valid], idx[pool_valid]


def _ref_select(q, w, k, gate, ape, topk, kpool, positions):
    """Per query row (at ``positions`` over keys ``k``): the set of token indices it attends."""
    keys, members = _ref_pools(k, gate, ape, kpool)
    out = []
    for r, p in enumerate(positions):
        visible = members[:, -1] <= p
        scores = torch.einsum("hd,jd->hj", q[r].float(), keys.float()).relu()
        scores = torch.einsum("h,hj->j", w[r].float(), scores)
        scores = scores.masked_fill(~visible, torch.finfo(scores.dtype).min)
        sel = scores.topk(min(topk // kpool, scores.shape[0])).indices
        tokens = {int(t) for j in sel if visible[j] for t in members[j]}
        n = p + 1
        tokens |= set(range(n - n % kpool, n))  # append_visible_tail
        out.append(tokens)
    return out


# -- a paged plane the helpers write and read, standing in for the KV manager -----------------


class _Planes:
    def __init__(self, num_planes, pages, page_size, width, dtype=torch.float32):
        self.cache = torch.zeros(num_planes, pages, page_size, width, dtype=dtype)
        self.slots = None  # this step's (page, offset) per row

    def layer_view(self, plane):
        return self.cache[plane]

    def write_kv(self, rows, v, layer_idx, label):
        page, off = self.slots
        self.cache[layer_idx][page, off] = rows.to(self.cache.dtype)


def _ctx(page_table, positions, page_size, topk, kpool, spans=None):
    """A one-request context over ``page_table`` for rows at ``positions``."""
    n = len(positions)
    return dsa.Glm5NextDsaContext(
        pos=torch.tensor(positions, dtype=torch.int32),
        row_req=torch.zeros(n, dtype=torch.int32),
        pages=torch.tensor(page_table, dtype=torch.int32),
        page_start=torch.zeros(1, dtype=torch.int32),
        host_pos=list(positions), spans=spans or [(0, n, 0)], host_page_start=[0],
        max_pools=(max(positions) + 1) // kpool, page_size=page_size, topk=topk, kpool=kpool)


def _slots_to_positions(slots, page_table, page_size):
    where = {page * page_size + o: i * page_size + o
             for i, page in enumerate(page_table) for o in range(page_size)}
    return {where[int(s)] for s in slots}


@pytest.mark.parametrize("seq,chunks", [(150, [150]), (150, [61, 1, 88]), (40, [40]),
                                        (67, [67]), (68, [65, 3])])
def test_selection_matches_reference(seq, chunks):
    torch.manual_seed(seq)
    nh, d, kpool, topk, page = 16, 16, 4, 64, 16
    k = torch.randn(seq, d).bfloat16().float()
    gate = torch.randn(seq, d)
    ape = torch.randn(kpool, d)
    q = torch.randn(seq, nh, d)
    w = torch.randn(seq, nh)
    pages = torch.randperm(40)[: -(-seq // page)].tolist()
    planes = _Planes(1, 40, page, 3 * d + 5)
    got = []
    start = 0
    for c in chunks:  # prefill steps that each write their tokens, then select
        positions = list(range(start, start + c))
        planes.slots = (torch.tensor([pages[p // page] for p in positions]),
                        torch.tensor([p % page for p in positions]))
        ctx = _ctx(pages, positions, page, topk, kpool)
        dsa.write_index(planes, 0, "main", k[start:start + c], gate[start:start + c], ape, ctx)
        slots = dsa.select(q[start:start + c], w[start:start + c], planes.layer_view(0), ctx)
        for r, p in enumerate(positions):
            n = dsa.attn_lens([p], topk, kpool)[0]
            got.append(_slots_to_positions(slots[r, :n].tolist(), pages, page))
        start += c
    ref = _ref_select(q, w, k, gate, ape, topk, kpool, list(range(seq)))
    assert got == ref
    # past the identity regime a row keeps exactly topk tokens plus its tail
    assert all(len(s) == min(p + 1, topk + (p + 1) % kpool) for p, s in enumerate(got))


class _FakeGroup:
    """A TP group of ``world`` ranks run one after another: each select call is one rank's,
    and all_gather returns the blocks every rank produced for that call."""

    def __init__(self, world):
        self.world_size, self.rank, self.blocks = world, 0, {}

    def all_gather(self, t, dim=0):
        self.blocks.setdefault(self.call, {})[self.rank] = t
        got = self.blocks[self.call]
        filler = torch.full_like(t, -7)
        return torch.cat([got.get(r, filler) for r in range(self.world_size)], dim=0)


def test_sharded_prefill_selection_equals_unsharded():
    """Each rank's block, gathered rank-major, is the unsharded selection."""
    torch.manual_seed(9)
    nh, d, kpool, topk, page, seq = 16, 16, 4, 32, 16, 333
    k, gate = torch.randn(seq, d), torch.randn(seq, d)
    ape, q, w = torch.randn(kpool, d), torch.randn(seq, nh, d), torch.randn(seq, nh)
    pages = list(range(-(-seq // page)))
    planes = _Planes(1, len(pages), page, 3 * d)
    planes.slots = (torch.tensor([p // page for p in range(seq)]),
                    torch.tensor([p % page for p in range(seq)]))
    ctx = _ctx(pages, list(range(seq)), page, topk, kpool)
    dsa.write_index(planes, 0, "main", k, gate, ape, ctx)
    want = dsa.select(q, w, planes.layer_view(0), ctx)
    group = _FakeGroup(4)
    group.call = 0
    outs = []
    for rank in range(4):  # the last rank's call sees every block
        group.rank = rank
        outs.append(dsa.select(q, w, planes.layer_view(0), ctx, group))
    assert torch.equal(outs[-1], want)


def test_pool_keys_match_reference_bf16():
    torch.manual_seed(0)
    d, kpool, page, seq = 16, 4, 8, 37
    k = torch.randn(seq, d).bfloat16()
    gate = torch.randn(seq, d).bfloat16()
    ape = torch.randn(kpool, d)
    pages = [3, 1, 4, 0, 2]
    planes = _Planes(1, 5, page, 3 * d, dtype=torch.bfloat16)
    positions = list(range(seq))
    planes.slots = (torch.tensor([pages[p // page] for p in positions]),
                    torch.tensor([p % page for p in positions]))
    ctx = _ctx(pages, positions, page, 64, kpool)
    dsa.write_index(planes, 0, "main", k, gate, ape, ctx)
    ref, members = _ref_pools(k, gate, ape, kpool)
    flat = planes.cache[0].view(-1, 3 * d)
    for j, last in enumerate(members[:, -1].tolist()):
        slot = pages[last // page] * page + last % page
        assert torch.equal(flat[slot, 2 * d:], ref[j])


def test_decode_rows_select_like_prefill_rows():
    """The decode path (one row per request, several requests) picks what the prefill path
    picks for the same rows."""
    torch.manual_seed(1)
    nh, d, kpool, topk, page, seq = 16, 16, 4, 32, 16, 120
    k, gate = torch.randn(seq, d), torch.randn(seq, d)
    ape, q, w = torch.randn(kpool, d), torch.randn(seq, nh, d), torch.randn(seq, nh)
    pages = list(range(8))
    planes = _Planes(1, 8, page, 3 * d)
    planes.slots = (torch.tensor([p // page for p in range(seq)]),
                    torch.tensor([p % page for p in range(seq)]))
    ctx = _ctx(pages, list(range(seq)), page, topk, kpool)
    dsa.write_index(planes, 0, "main", k, gate, ape, ctx)
    prefill = dsa.select(q, w, planes.layer_view(0), ctx)
    rows = [7, 64, 119, 33]
    dec = dsa.Glm5NextDsaContext(
        pos=torch.tensor(rows, dtype=torch.int32), row_req=torch.zeros(4, dtype=torch.int32),
        pages=torch.tensor(pages, dtype=torch.int32), page_start=torch.zeros(1, dtype=torch.int32),
        host_pos=rows, spans=[(i, 1, 0) for i in range(4)], host_page_start=[0],
        max_pools=(max(rows) + 1) // kpool, page_size=page, topk=topk, kpool=kpool)
    decode = dsa.select(q[rows], w[rows], planes.layer_view(0), dec)
    for i, r in enumerate(rows):
        n = dsa.attn_lens([r], topk, kpool)[0]
        assert sorted(decode[i, :n].tolist()) == sorted(prefill[r, :n].tolist())


# -- the long-context engine path on the reduced model ----------------------------------------


class _GreedySampler(Resource):
    @classmethod
    def build(cls, spec, info):
        return cls()

    def sample(self, request_ids, logits, **kwargs):
        return logits.argmax(-1)


class _StubTransfer:
    def __init__(self, *args, **kwargs):
        pass

    def get_kv_transfer_info(self, **kwargs):
        del kwargs

    def cleanup(self):
        pass

    def start_async_retrieve(self, **kwargs):
        del kwargs

    def owns_transfer_info(self, transfer_info, **kwargs):
        del transfer_info, kwargs
        return False

    def remove_request(self, request_id):
        del request_id


@pytest.fixture(autouse=True)
def _stubs(monkeypatch):
    fi = types.ModuleType("flashinfer")
    fi.norm = types.SimpleNamespace(
        rmsnorm=lambda x, w, eps=1e-6: (
            x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
        ).to(x.dtype) * w.to(x.dtype)
    )
    monkeypatch.setitem(sys.modules, "flashinfer", fi)
    monkeypatch.setattr(kv_mod, "KVTransferManager", _StubTransfer)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _engine(long_context: bool, window: int = 8192, seed: int = 0):
    torch.manual_seed(seed)
    cfg = Glm5NextModelConfig.reduced_long_context(max_seq_len=1024)
    cfg = dataclasses.replace(cfg, dsa_long_context=long_context, prefill_window_tokens=window)
    model = Glm5NextModel("x", config_variant="reduced", kda_conv_dtype=torch.float32,
                          kda_max_requests=4)
    model.config = cfg
    lm = Glm5NextForCausalLM(cfg)
    for p in lm.parameters():
        if p.dtype.is_floating_point:
            torch.nn.init.normal_(p, std=0.05)
    process_weights_after_loading(lm, torch.device("cpu"))
    lm.eval()
    sub = Glm5NextLLMSubmodule(lm, cfg)
    engine = Engine(None)
    engine.load_model(
        {"LLM": sub}, model.get_node_resources(),
        parallel_groups=WorkerParallelGroups(num_workers=1, global_rank=0),
        device=torch.device("cpu"),
        transfer_engine_info=TransferEngineInfo(
            my_entity_id="e", my_session_id="s",
            transfer_engine=LocalTransferEngine("localhost")),
        kv_cache_type=torch.float32,
    )
    greedy = _GreedySampler()
    engine._resources[SAMPLER] = greedy
    engine._submodules["LLM"].resources[SAMPLER] = greedy
    sub.node_resources[SAMPLER] = greedy
    engine.warmup()
    return engine, sub, cfg


def _info(rid, walk):
    return CurrentForwardPassInfo(
        request_id=f"wire-{rid}", rid_handle=rid, graph_walk=walk, fwd_index=0, random_seed=0,
        max_tokens=4096, resource_configs={SAMPLER: SamplingReqConfig(temperature=0.0)})


def _step(engine, walk, inputs):
    rids = list(inputs)
    infos = {rid: _info(rid, walk) for rid in rids}
    batch = ExecutingBatch(
        node_name="LLM",
        per_request_info=infos,
        per_request_input_tensors={rid: {"text_inputs": [ids]} for rid, ids in inputs.items()},
        # the loop counts the engine reads, as the worker builds them (main #351)
        per_request_input_metadata={
            rid: InputMetadata(dynamic_loop_iter_counts=dict(info.dynamic_loop_iter_counts))
            for rid, info in infos.items()},
        step_context=StepContext(request_ids=tuple(rids), graph_walk=walk, slot=0, capture=False),
    )
    engine.prepare_inputs(batch)
    assert not batch.failed_requests, batch.failed_requests
    out = engine.exec_and_postprocess(batch)
    assert batch.admit_error is None, batch.admit_error
    engine.finalize_batch(batch)
    return {rid: int(out.per_rid_outputs[rid]["new_token"][0]) for rid in rids}


def _greedy(engine, rid, prompt, steps, prefill_cut=None):
    """Prefill ``prompt[:cut]``, feed the rest one token a step, then decode ``steps`` greedily."""
    engine.add_request(rid, {SAMPLER: SamplingReqConfig(temperature=0.0)})
    cut = prefill_cut or len(prompt)
    tok = _step(engine, "prefill", {rid: prompt[:cut]})[rid]
    for t in prompt[cut:].tolist():
        tok = _step(engine, "decode", {rid: torch.tensor([t])})[rid]
    out = [tok]
    for _ in range(steps):
        tok = _step(engine, "decode", {rid: torch.tensor([tok])})[rid]
        out.append(tok)
    return out


def test_long_context_matches_dense_in_the_identity_regime():
    """Within topk / kpool complete pools DSA selects everything: same tokens as dense MLA."""
    prompt = torch.randint(0, 250, (40,))
    dense, _, cfg = _engine(long_context=False)
    sparse, _, _ = _engine(long_context=True)
    steps = cfg.index_topk - 40 - 1  # the dense engine refuses past index_topk
    assert _greedy(sparse, "a", prompt, steps) == _greedy(dense, "a", prompt, steps)


def test_long_prefill_equals_stepwise_and_windowed_prefill():
    """Past index_topk: one prefill, a short prefill fed the rest a token a step, and a prefill
    run in token windows all run the same per-row selection, so they emit the same tokens."""
    torch.manual_seed(5)
    prompt = torch.randint(0, 250, (300,))
    one, sub, _ = _engine(long_context=True)
    whole = _greedy(one, "a", prompt, 12)
    stepwise = _greedy(_engine(long_context=True)[0], "a", prompt, 12, prefill_cut=200)
    windowed = _greedy(_engine(long_context=True, window=37)[0], "a", prompt, 12)
    assert whole == stepwise == windowed
    assert sub.request_state("a").get("context", 0) == 300 + 12


@pytest.mark.parametrize("window", [8192, 64])
def test_two_long_requests_batch_like_one(window):
    """Batched (and, with small windows, cut mid-request) prefill emits what each alone does."""
    torch.manual_seed(6)
    a, b = torch.randint(0, 250, (190,)), torch.randint(0, 250, (150,))
    alone_a = _greedy(_engine(long_context=True)[0], "a", a, 6)
    alone_b = _greedy(_engine(long_context=True)[0], "b", b, 6)
    engine, _, _ = _engine(long_context=True, window=window)
    for rid in "ab":
        engine.add_request(rid, {SAMPLER: SamplingReqConfig(temperature=0.0)})
    toks = _step(engine, "prefill", {"a": a, "b": b})
    seq = {"a": [toks["a"]], "b": [toks["b"]]}
    for _ in range(6):
        toks = _step(engine, "decode", {r: torch.tensor([seq[r][-1]]) for r in "ab"})
        for r in "ab":
            seq[r].append(toks[r])
    assert seq["a"] == alone_a and seq["b"] == alone_b


def test_one_token_prompts_prefill_eager_in_long_mode():
    """A prefill step of one-token prompts plans KDA as decode: it runs as one window, and its
    tokens are dense mode's (the identity regime)."""
    want = {}
    for long_context in (False, True):
        engine, _, _ = _engine(long_context)
        for rid in ("a", "b"):
            engine.add_request(rid, {SAMPLER: SamplingReqConfig(temperature=0.0)})
        toks = _step(engine, "prefill", {"a": torch.tensor([7]), "b": torch.tensor([42])})
        out = [toks]
        for _ in range(3):
            toks = _step(engine, "decode", {rid: torch.tensor([t]) for rid, t in toks.items()})
            out.append(toks)
        want[long_context] = out
    assert want[True] == want[False]


def test_bind_refuses_a_page_size_pools_straddle():
    cfg = Glm5NextModelConfig.reduced_long_context()
    sub = object.__new__(Glm5NextLLMSubmodule)
    sub.config = cfg
    kv = types.SimpleNamespace(kv_cache=types.SimpleNamespace(page_size=cfg.index_kpool * 4 + 2))
    with pytest.raises(ValueError, match="multiple of index_kpool"):
        sub.bind_node_resources({KV_CACHE: kv})
