"""glm5_next MTP through the real ``Engine`` on CPU: greedy output with drafting on equals
greedy output with it off, whatever the drafts (the MTP layer's own, an oracle that is always
right, one that is right for a prefix), and every verify step keeps its whole block in the
KV cache (rejected rows as holes)."""
from __future__ import annotations

import sys
import types

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
from mstar.model.glm5_next.components.causal_lm import Glm5NextForCausalLM
from mstar.model.glm5_next.config import KDA_STATE, KV_CACHE, SAMPLER, Glm5NextModelConfig
from mstar.model.glm5_next.glm5_next_model import Glm5NextModel, process_weights_after_loading
from mstar.model.glm5_next.submodules import Glm5NextLLMSubmodule
from mstar.model.submodule_base import InputMetadata


def _batch(**kwargs) -> ExecutingBatch:
    """An ExecutingBatch as the worker builds it: the engine reads each request's
    loop counts from its input metadata (main #351), not from its forward-pass info."""
    kwargs.setdefault("per_request_input_metadata", {
        rid: InputMetadata(dynamic_loop_iter_counts=dict(info.dynamic_loop_iter_counts))
        for rid, info in kwargs["per_request_info"].items()
    })
    return ExecutingBatch(**kwargs)

# Large enough that a continuation depends on its context: at 0.02 the greedy token is a
# function of the last token alone, and neither the cache holes nor the KDA replay could move it.
_WEIGHT_STD = 0.1


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


def _build_engine(k: int, seed: int = 0):
    torch.manual_seed(seed)
    cfg = Glm5NextModelConfig.reduced()
    cfg.mtp_num_draft_tokens = k
    model = Glm5NextModel("x", config_variant="reduced", kda_conv_dtype=torch.float32,
                          kda_max_requests=4, mtp_num_draft_tokens=k)
    lm = Glm5NextForCausalLM(cfg)
    # the trunk's parameters come first, so both engines draw the same trunk (building
    # the MTP layer draws from the generator too, hence the reseed)
    torch.manual_seed(seed)
    for p in lm.parameters():
        if p.dtype.is_floating_point:
            torch.nn.init.normal_(p, std=_WEIGHT_STD)
    process_weights_after_loading(lm, torch.device("cpu"))
    lm.eval()
    sub = Glm5NextLLMSubmodule(lm, cfg)
    engine = Engine(None)  # no offload, so no runtime
    engine.load_model(
        {"LLM": sub}, model.get_node_resources(),
        parallel_groups=WorkerParallelGroups(num_workers=1, global_rank=0),
        device=torch.device("cpu"),
        transfer_engine_info=TransferEngineInfo(
            my_entity_id="e", my_session_id="s",
            transfer_engine=LocalTransferEngine("localhost"),
        ),
        kv_cache_type=torch.float32,
    )
    greedy = _GreedySampler()
    engine._resources[SAMPLER] = greedy
    engine._submodules["LLM"].resources[SAMPLER] = greedy
    sub.node_resources[SAMPLER] = greedy
    engine.warmup()
    return engine, sub, cfg


def _info(rid, walk, max_tokens, decode_iter=0, ignore_eos=False):
    return CurrentForwardPassInfo(
        request_id=f"wire-{rid}", rid_handle=rid, graph_walk=walk, fwd_index=0, random_seed=0, max_tokens=max_tokens,
        resource_configs={SAMPLER: SamplingReqConfig(temperature=0.0, ignore_eos=ignore_eos)},
        dynamic_loop_iter_counts={"decode_loop": decode_iter},
    )


def _step(engine, walk, inputs, max_tokens, decode_iter=0, cached=None, ignore_eos=False):
    rids = list(inputs)
    batch = _batch(
        node_name="LLM",
        per_request_info={rid: _info(rid, walk, max_tokens, decode_iter, ignore_eos)
                          for rid in rids},
        per_request_input_tensors={rid: {"text_inputs": [ids]} for rid, ids in inputs.items()},
        step_context=StepContext(request_ids=tuple(rids), graph_walk=walk, slot=0,
                                 capture=False),
    )
    engine.prepare_inputs(batch)
    assert not batch.failed_requests, batch.failed_requests
    # The cache holds the prompt and every step's whole block, rejected rows included (holes,
    # masked on the device), and the request's context counts them.
    kv, sub = engine._resources[KV_CACHE], engine._submodules["LLM"].submodule
    for rid, want in (cached or {}).items():
        assert kv._streams[rid]["main"].stored_len == want, (rid, want)
        assert sub.request_state(rid).get("context") == want, (rid, want)
    outputs = engine.exec_and_postprocess(batch)
    assert batch.admit_error is None, batch.admit_error
    engine.finalize_batch(batch)
    stops = engine.check_stop_for_batch(batch, outputs)
    return outputs.per_rid_outputs, stops


def _generate(engine, prompts: dict[str, torch.Tensor], max_tokens: int) -> dict[str, list]:
    for rid in prompts:
        engine.add_request(rid, {SAMPLER: SamplingReqConfig(temperature=0.0)})
    out, stops = _step(engine, "prefill", prompts, max_tokens)
    seqs = {rid: out[rid]["new_token"][0].tolist() for rid in prompts}
    nxt = {rid: out[rid]["text_inputs"][0] for rid in prompts}
    live = [rid for rid in prompts if not stops.get(rid)]
    block = engine._submodules["LLM"].submodule.mtp_block
    step = 0
    while live:
        cached = {rid: len(prompts[rid]) + block * step for rid in live}
        out, stops = _step(engine, "decode", {rid: nxt[rid] for rid in live}, max_tokens,
                           step, cached)
        step += 1
        for rid in live:
            seqs[rid] += out[rid]["new_token"][0].tolist()
            nxt[rid] = out[rid]["text_inputs"][0]
        live = [rid for rid in live if not stops.get(rid)]
    for rid in prompts:
        engine.remove_request(rid)
    pool = engine._resources[KDA_STATE]
    assert pool.num_free_slots == pool.config.usable_slots
    return seqs


PROMPTS = {"a": torch.tensor([5, 9, 13, 7, 21, 2]), "b": torch.tensor([44, 3, 8])}
MAX_TOKENS = 14


@pytest.fixture  # not module-scoped: it must build after _stubs hides CUDA
def reference():
    engine, _, _ = _build_engine(0)
    return _generate(engine, dict(PROMPTS), MAX_TOKENS)


def _oracle(sub, reference, wrong_every: int | None):
    """Drafts from the reference continuation at each request's position, wrong at every
    ``wrong_every``-th call when given. A step drafts d2..dk in its draft loop, then the next
    step's d1 once its verdict is in (the prefill: the first d1); wrapping the two passes tells
    the oracle which call is which and how far each request is."""
    calls = {"n": 0}
    where: dict = {}
    given: dict[str, list[int]] = {}  # each request's drafts for its current step, d1 first
    real_prefill, real_decode = sub._mtp_prefill, sub._mtp_decode

    def at(rid, pos):
        seq = reference[rid]
        return seq[pos] if pos < len(seq) else 0

    def emit(rids, tokens, fresh):
        calls["n"] += 1
        if wrong_every and calls["n"] % wrong_every == 0:
            tokens = [(t + 1) % 256 for t in tokens]
        for rid, t in zip(rids, tokens, strict=True):
            given[rid] = [t] if fresh else [*given[rid], t]
        return torch.tensor(tokens, dtype=torch.long)

    def prefill(engine_inputs, *args):
        where.update(rids=list(engine_inputs.request_ids), row=None)
        return real_prefill(engine_inputs, *args)

    def decode(engine_inputs, *args):
        rids = list(engine_inputs.request_ids)
        # tokens emitted before this step: its first draft is the one at that position
        where.update(rids=rids, row=0,
                     done={rid: sub.request_state(rid).get("mtp_generated") for rid in rids})
        return real_decode(engine_inputs, *args)

    def draft(hidden, prev):
        rids = where["rids"]
        if where["row"] is None:
            return emit(rids, [at(rid, 1) for rid in rids], fresh=True)
        done = where["done"]
        if where["row"] < sub.mtp_k - 1:
            where["row"] += 1
            return emit(rids, [at(rid, done[rid] + where["row"]) for rid in rids], fresh=False)
        # the verdict is in: the next step's d1 follows the accepted drafts and the bonus
        nxt = []
        for rid in rids:
            accepted = 0
            while accepted < sub.mtp_k and given[rid][accepted] == at(rid, done[rid] + accepted):
                accepted += 1
            nxt.append(at(rid, done[rid] + accepted + 1))
        return emit(rids, nxt, fresh=True)

    sub._mtp_prefill, sub._mtp_decode, sub._draft_tokens = prefill, decode, draft


def test_reference_depends_on_context(reference):
    """Some token is followed by different tokens: a draft keyed on the last token alone
    would fail, and so would a verify step that attends a hole or replays the wrong KDA
    prefix."""
    assert all(len(seq) == MAX_TOKENS for seq in reference.values()), reference
    successors: dict[int, set[int]] = {}
    for seq in reference.values():
        for a, b in zip(seq, seq[1:], strict=False):
            successors.setdefault(a, set()).add(b)
    assert any(len(b) > 1 for b in successors.values()), reference


@pytest.mark.parametrize("k", [1, 3])
@pytest.mark.parametrize("drafts", ["mtp", "oracle", "oracle_partial"])
def test_mtp_output_equals_greedy_without_mtp(reference, k, drafts):
    engine, sub, _ = _build_engine(k)
    if drafts != "mtp":
        _oracle(sub, reference, wrong_every=3 if drafts == "oracle_partial" else None)
    got = _generate(engine, dict(PROMPTS), MAX_TOKENS)
    assert got == reference


def test_oracle_drafts_are_all_accepted():
    """With always-right drafts every step emits k + 1 tokens."""
    k = 3
    engine, sub, _ = _build_engine(k)
    ref_engine, _, _ = _build_engine(0)
    reference = _generate(ref_engine, {"a": PROMPTS["a"]}, MAX_TOKENS)
    _oracle(sub, reference, wrong_every=None)
    engine.add_request("a", {SAMPLER: SamplingReqConfig(temperature=0.0)})
    out, _ = _step(engine, "prefill", {"a": PROMPTS["a"]}, MAX_TOKENS)
    first = out["a"]["new_token"][0].tolist()
    for step in range(2):
        out, _ = _step(engine, "decode", {"a": out["a"]["text_inputs"][0]}, MAX_TOKENS, step)
        emitted = out["a"]["new_token"][0].tolist()
        assert len(emitted) == k + 1
        assert first + emitted == reference["a"][:len(first) + len(emitted)]
        first += emitted


def test_decode_stops_while_the_next_block_fits():
    """The context counts emitted tokens against index_topk (the cache's holes don't):
    decode stops while the step that may already be scheduled still fits, and that step
    runs."""
    k = 3
    engine, sub, cfg = _build_engine(k)
    block, limit = k + 1, cfg.index_topk
    prompt = torch.arange(2, limit - 2 * block - 9)
    engine.add_request("r", {SAMPLER: SamplingReqConfig(temperature=0.0)})
    out, stops = _step(engine, "prefill", {"r": prompt}, 10 * limit, ignore_eos=True)
    step = 0
    while not stops.get("r"):
        out, stops = _step(engine, "decode", {"r": out["r"]["text_inputs"][0]}, 10 * limit,
                           step, ignore_eos=True)
        step += 1
    real = len(prompt) + sub.request_state("r").get("mtp_generated")
    assert real + block <= limit < real + 2 * block
    _step(engine, "decode", {"r": out["r"]["text_inputs"][0]}, 10 * limit, step,
          ignore_eos=True)
    assert sub.request_state("r").get("context") <= cfg.kv_rows
    engine.remove_request("r")


def _stop_after(prompt_len, max_tokens, per_step, k=3, index_topk=2048, kv_rows=4096):
    """(verify steps, tokens emitted) when _mtp_check_stop ends a request that emits
    ``per_step`` tokens per step, ignore_eos."""
    from mstar.model.submodule_base import PerRequestState

    state = PerRequestState()
    state.add("prompt_len", prompt_len)
    sub = types.SimpleNamespace(
        request_state=lambda rid: state, mtp_block=k + 1,
        config=types.SimpleNamespace(eos_token_ids=(), index_topk=index_topk, kv_rows=kv_rows),
    )
    sampler = {SAMPLER: types.SimpleNamespace(ignore_eos=True)}

    def stops(walk, steps, tokens):
        info = types.SimpleNamespace(graph_walk=walk, resource_configs=sampler,
                                     dynamic_loop_iter_counts={"decode_loop": steps},
                                     max_tokens=max_tokens)
        return Glm5NextLLMSubmodule._mtp_check_stop(sub, "r", info, torch.tensor(tokens))

    if stops("prefill", 0, [7]):
        return 0, 1
    generated = 1
    for step in range(1, 100_000):
        generated += per_step
        state.add("mtp_generated", generated)
        if stops("decode", step - 1, [7] * per_step):
            return step, generated
    raise AssertionError("never stopped")


@pytest.mark.parametrize("per_step", [1, 2, 3])
def test_rejected_drafts_do_not_cut_the_output_short(per_step):
    # 1024 in / 512 out: every token arrives however few drafts are accepted
    assert _stop_after(1024, 512, per_step)[1] >= 512


def test_rejected_drafts_stop_at_the_cache_rows():
    # a short prompt with no draft accepted fills the cache's rows before the context
    steps, generated = _stop_after(16, 100_000, 1)
    rows = 16 + steps * 4
    assert rows <= 4096 < rows + 2 * 4
    assert 16 + generated < 2048


def test_acceptance_is_logged(caplog):
    engine, sub, _ = _build_engine(3)
    sub._mtp_logged = float("-inf")  # the first step logs
    with caplog.at_level("INFO", logger="mstar.model.glm5_next.submodules"):
        _generate(engine, {"a": PROMPTS["a"]}, 4)
    assert any("drafts accepted per step" in r.getMessage() for r in caplog.records)


def test_temperature_is_refused_under_mtp():
    engine, _, _ = _build_engine(2)
    engine.add_request("t", {SAMPLER: SamplingReqConfig(temperature=0.7)})
    batch = _batch(
        node_name="LLM",
        per_request_info={"t": CurrentForwardPassInfo(
            request_id="wire-t", rid_handle="t", graph_walk="prefill", fwd_index=0, random_seed=0,
            max_tokens=4,
            resource_configs={SAMPLER: SamplingReqConfig(temperature=0.7)})},
        per_request_input_tensors={"t": {"text_inputs": [PROMPTS["b"]]}},
        step_context=StepContext(request_ids=("t",), graph_walk="prefill", slot=0,
                                 capture=False),
    )
    engine.prepare_inputs(batch)
    assert "greedy only" in str(batch.failed_requests.get("t"))


def test_prompts_leave_two_blocks_of_index_topk():
    model = Glm5NextModel("x", config_variant="reduced", tokenizer_mode="byte",
                          mtp_num_draft_tokens=3)
    limit = model.config.index_topk - 2 * 4
    assert model.process_prompt("a" * limit, ["text"], ["text"])["text_inputs"][0].numel() == limit
    with pytest.raises(ValueError, match="at most"):
        model.process_prompt("a" * (limit + 1), ["text"], ["text"])


def test_vocab_parallel_argmax_equals_the_full_argmax():
    """``_argmax`` over 4 emulated TP ranks, ties across shards included, against the argmax
    of the gathered logits."""
    from types import SimpleNamespace

    torch.manual_seed(0)
    tp, vocab, hidden = 4, 64, 16
    weight = torch.randn(vocab, hidden)
    x = torch.randn(9, hidden)
    weight[5] = weight[21]  # rows 5 and 21 score the same: row 21's tie goes to shard 0
    x[0] = weight[21] * 10
    shard = vocab // tp
    locals_ = [torch.nn.functional.linear(x, weight[r * shard:(r + 1) * shard]) for r in range(tp)]

    class _Comm:
        def __init__(self):
            self.calls = 0

        def all_gather(self, t, dim=-1):
            self.calls += 1
            per_rank = [loc.gather(-1, loc.argmax(-1, keepdim=True)).float() if self.calls == 1
                        else loc.argmax(-1, keepdim=True) + r * shard
                        for r, loc in enumerate(locals_)]
            return torch.cat(per_rank, dim=dim)

    full = torch.nn.functional.linear(x, weight).argmax(-1)
    for rank in range(tp):
        head = SimpleNamespace(weight=weight[rank * shard:(rank + 1) * shard], tp_size=tp,
                               tp_rank=rank, output_size_per_partition=shard,
                               comm_group=_Comm())
        got = Glm5NextLLMSubmodule._argmax(SimpleNamespace(lm_head=head), x)
        assert torch.equal(got, full), (rank, got, full)
    assert full[0] == 5


def test_a_padding_row_writes_its_real_rows_token():
    """A captured prefill's padding row repeats the previous row's last index: the MTP input
    there is the real row's sampled token, not the padding row's (sampled at slot 0's
    temperature)."""
    from mstar.model.glm5_next.submodules import _real_row_values
    from mstar.model.submodule_base import ARNodeInputs

    sub = types.SimpleNamespace(lm_head=types.SimpleNamespace(weight=torch.zeros(1)))
    rows = [ARNodeInputs(input_ids=torch.zeros(n, dtype=torch.long), input_seq_len=n)
            for n in (5, 3, 4, 0)]
    last = Glm5NextLLMSubmodule._last_rows(sub, rows).long()
    nxt = torch.arange(12)
    nxt.index_copy_(0, last, _real_row_values(last, torch.tensor([100, 101, 102, 999])))
    assert nxt[last].tolist() == [100, 101, 102, 102]


def test_the_stop_counts_this_steps_tokens_not_the_next_ones():
    """The next step is in flight when this one is checked, and the GPU thread's count may
    already include its tokens. 1 + 4 + 4 tokens reach max_tokens=9 at the second verify
    step, not the first."""
    from mstar.model.submodule_base import PerRequestState

    state = PerRequestState()
    state.add("prompt_len", 16)
    sub = types.SimpleNamespace(
        request_state=lambda rid: state, mtp_block=4,
        config=types.SimpleNamespace(eos_token_ids=(), index_topk=2048, kv_rows=4096),
    )
    sampler = {SAMPLER: types.SimpleNamespace(ignore_eos=True)}

    def check(walk, steps, n):
        info = types.SimpleNamespace(graph_walk=walk, resource_configs=sampler,
                                     dynamic_loop_iter_counts={"decode_loop": steps},
                                     max_tokens=9)
        return Glm5NextLLMSubmodule._mtp_check_stop(sub, "r", info, torch.tensor([7] * n))

    state.add("mtp_generated", 5)  # the first verify step already unpacked
    assert not check("prefill", 0, 1)
    state.add("mtp_generated", 9)  # and the second
    assert not check("decode", 0, 4)
    assert check("decode", 1, 4)
