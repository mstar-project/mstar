"""glm5_next MTP through the real ``Engine`` on CPU: greedy output with drafting on equals
greedy output with it off, whatever the drafts (the MTP layer's own, an oracle that is always
right, one that is right for a prefix), and every verify step keeps its whole block in the
KV cache (rejected rows as holes)."""
from __future__ import annotations

import logging
import sys
import threading
import types

import pytest
import torch
from request_walks import run_walks

from mstar.communication.tensors import LocalTransferEngine
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.communication import WorkerParallelGroups
from mstar.engine.engine import Engine, ExecutingBatch
from mstar.engine.resources import SamplingReqConfig
from mstar.engine.resources.kv import manager as kv_mod
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.engine.resources.step import StepContext
from mstar.model.base import serving_resources
from mstar.model.glm5_next.components.causal_lm import Glm5NextForCausalLM
from mstar.model.glm5_next.config import COUNT, KDA_STATE, KV_CACHE, LABEL, MTP_SEED, PROMPT_LEN, SAMPLER
from mstar.model.glm5_next.glm5_next_model import Glm5NextModel, process_weights_after_loading
from mstar.model.glm5_next.submodules import Glm5NextLLMSubmodule
from mstar.model.submodule_base import ARNodeInputs, InputMetadata
from mstar.testing_utils.resource import StubResource
from mstar.utils.device import DeviceCapability
from mstar.worker.host_copies import HostCopies


def _batch(**kwargs) -> ExecutingBatch:
    """An ExecutingBatch as the worker builds it, each request's loop counts in its input
    metadata."""
    kwargs.setdefault("per_request_input_metadata", {
        rid: InputMetadata(dynamic_loop_iter_counts=dict(info.dynamic_loop_iter_counts))
        for rid, info in kwargs["per_request_info"].items()
    })
    return ExecutingBatch(**kwargs)

# Large enough that a continuation depends on its context: at 0.02 the greedy token is a
# function of the last token alone, and neither the cache holes nor the KDA replay could move it.
_WEIGHT_STD = 0.1


class _GreedySampler(StubResource):
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
    model = Glm5NextModel("x", config_variant="reduced", kda_conv_dtype=torch.float32,
                          kda_max_requests=4, mtp_num_draft_tokens=k)
    cfg = model.config
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
        {"LLM": sub}, serving_resources(model, {}),
        parallel_groups=WorkerParallelGroups(num_workers=1, global_rank=0),
        device=torch.device("cpu"),
        transfer_engine_info=TransferEngineInfo(
            my_entity_id="e", my_session_id="s",
            transfer_engine=LocalTransferEngine("localhost"),
        ),
        kv_cache_type=torch.float32,
        model=model,
    )
    greedy = _GreedySampler()
    engine._resources[SAMPLER] = greedy
    engine._submodules["LLM"].resources[SAMPLER] = greedy
    sub.node_resources[SAMPLER] = greedy
    sub.sampler.resource = greedy
    engine.warmup()
    return engine, sub, cfg


def _info(rid, walk, max_tokens, decode_iter=0, ignore_eos=False, prompt_len=0):
    return CurrentForwardPassInfo(
        request_id=f"wire-{rid}", rid_handle=rid, graph_walk=walk, fwd_index=0, random_seed=0, max_tokens=max_tokens,
        resource_configs={SAMPLER: SamplingReqConfig(temperature=0.0, ignore_eos=ignore_eos)},
        dynamic_loop_iter_counts={"decode_loop": decode_iter},
        # the walk plan puts the prompt's length in the decode step's metadata alone
        step_metadata={PROMPT_LEN: prompt_len} if walk == "decode" else {},
    )


def _carry(out: dict) -> dict:
    """The decode loop's inputs from a step's outputs: the next token, and under MTP the count."""
    return {name: out[name] for name in ("text_inputs", COUNT) if name in out}


def _stored(engine, rid: str) -> int:
    return engine._resources[KV_CACHE]._streams[rid][LABEL].stored_len


def _step(engine, walk, inputs, max_tokens, decode_iter=0, cached=None, ignore_eos=False, prompt_lens=None):
    """One step of ``walk``; ``inputs`` holds each request's input tensors by name."""
    rids = list(inputs)
    batch = _batch(
        node_name="LLM",
        per_request_info={rid: _info(rid, walk, max_tokens, decode_iter, ignore_eos, (prompt_lens or {}).get(rid, 0))
                          for rid in rids},
        per_request_input_tensors=inputs,
        step_context=StepContext(request_ids=tuple(rids), graph_walk=walk, slot=0,
                                 capture=False),
    )
    engine.prepare_inputs(batch)
    assert not batch.failed_requests, batch.failed_requests
    # The cache holds the prompt and every step's whole block, rejected rows included (holes,
    # masked on the device).
    for rid, want in (cached or {}).items():
        assert _stored(engine, rid) == want, (rid, want)
    outputs = engine.exec_and_postprocess(batch)
    assert batch.admit_error is None, batch.admit_error
    engine.finalize_batch(batch)
    # as the worker hands the stop check a step's outputs: its stop buffers where the forward set them
    cpu = torch.device("cpu")
    copies = HostCopies(device=cpu, device_capability=DeviceCapability.of(cpu))
    host, rows = copies.prematerialize_for_check_stop(outputs, None, request_ids=batch.request_ids)
    stops = engine.check_stop_for_batch(batch, host, host_rows=rows)
    assert not batch.failed_requests, batch.failed_requests
    return outputs.per_rid_outputs, stops


def _generate(engine, prompts: dict[str, torch.Tensor], max_tokens: int) -> dict[str, list]:
    for rid in prompts:
        engine.add_request(rid, {SAMPLER: SamplingReqConfig(temperature=0.0)})
    lens = {rid: len(prompt) for rid, prompt in prompts.items()}
    out, stops = _step(engine, "prefill", {rid: {"text_inputs": [p]} for rid, p in prompts.items()}, max_tokens)
    seqs = {rid: out[rid]["new_token"][0].tolist() for rid in prompts}
    nxt = {rid: _carry(out[rid]) for rid in prompts}
    live = [rid for rid in prompts if not stops.get(rid)]
    block = engine._submodules["LLM"].submodule.mtp_block
    step = 0
    while live:
        cached = {rid: lens[rid] + block * step for rid in live}
        out, stops = _step(engine, "decode", {rid: nxt[rid] for rid in live}, max_tokens,
                           step, cached, prompt_lens=lens)
        step += 1
        for rid in live:
            seqs[rid] += out[rid]["new_token"][0].tolist()
            nxt[rid] = _carry(out[rid])
        live = [rid for rid in live if not stops.get(rid)]
    for rid in prompts:
        engine.remove_request(rid)
    for key in (KDA_STATE, MTP_SEED) if block > 1 else (KDA_STATE,):
        pool = engine._resources[key]
        assert pool.num_free_slots == pool.config.usable_slots, f"{key} kept a slot of a removed request"
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
    the oracle which call is which, and the verdicts how far each request is."""
    calls = {"n": 0}
    where: dict = {}
    given: dict[str, list[int]] = {}  # each request's drafts for its current step, d1 first
    done: dict[str, int] = {}  # each request's tokens emitted before its current step
    real_forward, real_decode = sub._forward, sub._mtp_decode

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

    def prefill(graph_walk, engine_inputs, *args):
        if graph_walk == "prefill":
            where.update(rids=list(engine_inputs.request_ids), row=None)
            done.update(dict.fromkeys(engine_inputs.request_ids, 1))
        return real_forward(graph_walk, engine_inputs, *args)

    def decode(engine_inputs, *args):
        where.update(rids=list(engine_inputs.request_ids), row=0)
        return real_decode(engine_inputs, *args)

    def draft(hidden, prev):
        rids = where["rids"]
        if where["row"] is None:
            return emit(rids, [at(rid, 1) for rid in rids], fresh=True)
        if where["row"] < sub.mtp_k - 1:
            where["row"] += 1
            return emit(rids, [at(rid, done[rid] + where["row"]) for rid in rids], fresh=False)
        # the verdict is in: the next step's d1 follows the accepted drafts and the bonus
        nxt = []
        for rid in rids:
            accepted = 0
            while accepted < sub.mtp_k and given[rid][accepted] == at(rid, done[rid] + accepted):
                accepted += 1
            done[rid] += accepted + 1
            nxt.append(at(rid, done[rid]))
        return emit(rids, nxt, fresh=True)

    sub._forward, sub._mtp_decode, sub._draft_tokens = prefill, decode, draft


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
    out, _ = _step(engine, "prefill", {"a": {"text_inputs": [PROMPTS["a"]]}}, MAX_TOKENS)
    first = out["a"]["new_token"][0].tolist()
    for step in range(2):
        out, _ = _step(engine, "decode", {"a": _carry(out["a"])}, MAX_TOKENS, step,
                       prompt_lens={"a": len(PROMPTS["a"])})
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
    lens = {"r": len(prompt)}
    out, stops = _step(engine, "prefill", {"r": {"text_inputs": [prompt]}}, 10 * limit, ignore_eos=True)
    step = 0
    while not stops.get("r"):
        out, stops = _step(engine, "decode", {"r": _carry(out["r"])}, 10 * limit,
                           step, ignore_eos=True, prompt_lens=lens)
        step += 1
    real = len(prompt) + int(out["r"][COUNT][0][0])
    assert real + block <= limit < real + 2 * block
    _step(engine, "decode", {"r": _carry(out["r"])}, 10 * limit, step, ignore_eos=True, prompt_lens=lens)
    assert _stored(engine, "r") == len(prompt) + block * (step + 1), "the step scheduled before the stop didn't run"
    engine.remove_request("r")


def _stop_after(prompt_len, max_tokens, per_step, k=3, context_limit=2048):
    """(verify steps, tokens emitted) when _mtp_stops ends a request that emits
    ``per_step`` tokens per step, ignore_eos."""
    sub = types.SimpleNamespace(
        mtp_block=k + 1, config=types.SimpleNamespace(eos_token_ids=(), context_limit=context_limit),
    )
    generated = 1
    for step in range(1, 100_000):
        generated += per_step
        if Glm5NextLLMSubmodule._mtp_stops(sub, False, generated, max_tokens, prompt_len):
            return step, generated
    raise AssertionError("never stopped")


@pytest.mark.parametrize("per_step", [1, 2, 3])
def test_rejected_drafts_do_not_cut_the_output_short(per_step):
    # 1024 in / 512 out: every token arrives however few drafts are accepted
    assert _stop_after(1024, 512, per_step)[1] >= 512


def test_with_no_draft_accepted_the_context_stops_inside_the_holed_rows():
    """Every verify step keeps its whole block, and the holed pool keeps k + 1 rows a token of
    the context, so the context's stop comes first."""
    steps, generated = _stop_after(16, 100_000, 1)
    assert 16 + generated + 4 <= 2048 < 16 + generated + 2 * 4, "the context stops while the next block fits"
    assert 16 + (steps + 1) * 4 <= 2048 * 4, "the steps run and the one in flight overflow the pool's rows"


def test_vocab_parallel_argmax_equals_the_full_argmax():
    """``_argmax`` over 4 emulated TP ranks, ties across shards included, against the argmax
    of the gathered logits; one gather of the ranks' (logit, id) pairs per call."""
    from types import SimpleNamespace

    from mstar.model.glm5_next.submodules import _shard_best

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
            per_rank = [_shard_best(loc, r * shard) for r, loc in enumerate(locals_)]
            assert t.dtype == per_rank[0].dtype and t.shape == per_rank[0].shape
            return torch.cat(per_rank, dim=dim)

    full = torch.nn.functional.linear(x, weight).argmax(-1)
    for rank in range(tp):
        comm = _Comm()
        head = SimpleNamespace(weight=weight[rank * shard:(rank + 1) * shard], tp_size=tp,
                               tp_rank=rank, output_size_per_partition=shard, comm_group=comm)
        got = Glm5NextLLMSubmodule._argmax(SimpleNamespace(lm_head=head), x)
        assert torch.equal(got, full), (rank, got, full)
        assert got.dtype == torch.int64 and comm.calls == 1
    assert full[0] == 5


def test_a_padding_row_writes_its_real_rows_token():
    """A captured prefill's padding row repeats the previous row's last index: the MTP input
    there is the real row's sampled token, not the padding row's (sampled at slot 0's
    temperature)."""
    from mstar.model.glm5_next.submodules import _real_row_values
    from mstar.model.submodule_base import ARNodeInputs
    from mstar.utils.h2d import PinnedStager

    sub = types.SimpleNamespace(lm_head=types.SimpleNamespace(weight=torch.zeros(1)),
                                _int_stager=PinnedStager(torch.int32))
    rows = [ARNodeInputs(input_ids=torch.zeros(n, dtype=torch.long), input_seq_len=n)
            for n in (5, 3, 4, 0)]
    last = Glm5NextLLMSubmodule._last_rows(sub, rows).long()
    nxt = torch.arange(12)
    nxt.index_copy_(0, last, _real_row_values(last, torch.tensor([100, 101, 102, 999])))
    assert nxt[last].tolist() == [100, 101, 102, 102]


# -- a stop on the prefill walk -----------------------------------------------


def _serve(engine, prompt: torch.Tensor, max_tokens: int, ignore_eos=False, walks=None) -> list[int]:
    """The tokens one request sends the client, its stops going through the graph runtime;
    ``walks`` collects the walk of each step it ran."""
    engine.add_request("r", {SAMPLER: SamplingReqConfig(temperature=0.0, ignore_eos=ignore_eos)})

    def step(walk, inputs, iteration):
        if walks is not None:
            walks.append(walk)
        out, stops = _step(engine, walk, {"r": inputs}, max_tokens, iteration, ignore_eos=ignore_eos,
                           prompt_lens={"r": len(prompt)})
        return out.get("r", {}), stops.get("r", set())

    k = engine._submodules["LLM"].submodule.mtp_k
    tokens = run_walks(Glm5NextModel("x", config_variant="reduced", mtp_num_draft_tokens=k),
                       step, prompt, max_tokens, model_kwargs={PROMPT_LEN: len(prompt)})
    engine.remove_request("r")
    return tokens


def _errors(caplog) -> list[str]:
    # On the CPU the MLA resource runs its fallback, which no graph replays, so the runner drops the decode capture.
    dropped = "Dropping accelerator graph config 0 for LLM: resources cannot capture: ['attn']"
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR and r.getMessage() != dropped]


@pytest.mark.parametrize("k", [0, 3])
def test_a_stop_token_from_the_prefill_ends_the_reply(k, caplog):
    # the prefill walk has no decode_loop: its stop ends the partition, and no decode step runs
    engine, _, cfg = _build_engine(k)
    eos = cfg.eos_token_ids[1]
    greedy = engine._resources[SAMPLER]

    def eos_once(rids, logits, **kw):
        del greedy.sample  # back to the class's argmax
        return torch.full((logits.shape[0],), eos, dtype=torch.long)

    greedy.sample = eos_once
    walks = []
    assert _serve(engine, PROMPTS["a"], MAX_TOKENS, walks=walks) == [eos]
    assert walks == ["prefill"]
    assert _errors(caplog) == []


@pytest.mark.parametrize("k", [0, 3])
def test_a_prefill_stop_check_needs_only_the_rows_the_worker_copies(k):
    """The worker hands a prefill's stop check the step's stop buffers, its sampled tokens alone, and not the count
    the prefill persists for the decode loop. A check that read the count failed every request at its prefill, an
    HTTP 500 under ``mstar serve`` at TP2, where handing check_stop the engine's own outputs passed."""
    engine, _, _ = _build_engine(k)
    engine.add_request("r", {SAMPLER: SamplingReqConfig(temperature=0.0)})
    out, _ = _step(engine, "prefill", {"r": {"text_inputs": [PROMPTS["a"]]}}, MAX_TOKENS)
    assert out["r"]["new_token"][0].numel() == 1, "the prefill emits one token"


@pytest.mark.parametrize("k", [0, 3])
@pytest.mark.parametrize("max_tokens", [1, 2, 3])
def test_short_budgets_are_exact_and_name_no_missing_loop(k, max_tokens, caplog):
    engine, _, _ = _build_engine(k)
    assert len(_serve(engine, PROMPTS["a"], max_tokens, ignore_eos=True)) == max_tokens
    assert _errors(caplog) == []


# -- a verdict that never lands -------------------------------------------------


def _verdict_reader(monkeypatch, verdicts=((0, 0, 0, 0, 0),), ready=0):
    """A submodule whose verify step in flight staged ``verdicts`` and the flag ``ready``: 0 is
    a step that never staged its verdict, as after a failed replay."""
    from mstar.model.glm5_next import submodules

    monkeypatch.setattr(submodules, "_VERDICT_DEADLINE_S", 0.2, raising=False)
    sub = Glm5NextLLMSubmodule.__new__(Glm5NextLLMSubmodule)
    staged = submodules._Mailbox(torch.tensor(verdicts, dtype=torch.long), torch.tensor([ready]))
    config = types.SimpleNamespace(eos_token_ids=(), context_limit=2048)
    sub.__dict__.update(mtp_k=3, _staged=staged, config=config, _log_acceptance=lambda accepted: None)
    return sub


def _raised(fn, seconds: float = 20.0):
    """What ``fn`` raised, on a thread bounded at ``seconds``: a wait with no deadline fails here."""
    out = {}

    def run():
        try:
            fn()
        except BaseException as exc:  # noqa: BLE001 - SystemExit is the point
            out["exc"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(seconds)
    assert not thread.is_alive(), "the verdict read is still waiting"
    return out.get("exc")


def test_a_verdict_that_never_lands_exits_the_process(monkeypatch):
    """A step that failed before its verdict landed: the read exits the process at its deadline
    instead of holding the worker forever."""
    sub = _verdict_reader(monkeypatch)
    exc = _raised(lambda: sub._read_verdict(1))
    assert isinstance(exc, SystemExit) and "never staged" in str(exc)


def test_a_verify_step_carries_its_count_to_the_stop(monkeypatch):
    """The prefill emits a count of 1, and a verify step the count it came in with plus the
    tokens it emits, cut at max_tokens, which its stop reads: 1 + 4 + 4 tokens reach
    max_tokens=9 at the second verify step, and a step that comes in at 7 emits 2."""
    sub = _verdict_reader(monkeypatch, verdicts=((5, 6, 7, 8, 3),), ready=1)
    sampler = {SAMPLER: types.SimpleNamespace(ignore_eos=True)}
    prefill = types.SimpleNamespace(graph_walk="prefill", max_tokens=9, resource_configs=sampler, step_metadata={})
    decode = types.SimpleNamespace(graph_walk="decode", max_tokens=9, resource_configs=sampler,
                                   step_metadata={PROMPT_LEN: 16})
    (out,) = sub.unpack_packed_outputs({}, ["r"], [5], [ARNodeInputs(input_seq_len=5)], {"r": prefill}).values()
    assert out[COUNT][0].tolist() == [1, -1, 9, True, -1]
    for count, emitted, stops in ((1, 4, set()), (5, 4, {"decode_loop"}), (7, 2, {"decode_loop"})):
        kwargs = {"emitted": count, "max_tokens": 9, "ignore_eos": True, "prompt_len": 16}
        inputs = [ARNodeInputs(input_seq_len=4, kwargs=kwargs)]
        (out,) = sub.unpack_packed_outputs({}, ["r"], [4], inputs, {"r": decode}).values()
        assert out["new_token"][0].tolist() == [5, 6, 7, 8][:emitted]
        assert out[COUNT][0].tolist() == [count + emitted, bool(stops), 9, True, 16]
        assert sub.check_stop("r", decode, out) == stops, f"a step that came in at {count}"
