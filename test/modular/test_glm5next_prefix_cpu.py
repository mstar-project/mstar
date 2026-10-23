"""glm5_next prefix reuse through the real ``Engine`` on CPU: a prompt that shares a long prefix
with an earlier one skips it (its MLA pages from the KV index, its KDA state from a snapshot the
earlier prefill saved at a page boundary) and generates exactly what a fresh prefill of the
whole prompt does, with MTP and without."""
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
from mstar.engine.resources.kv.config import KVReqConfig, PagedKVConfig, PrefixSpan
from mstar.engine.resources.kv.keys import chain
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.engine.resources.recurrent.config import RecurrentStateConfig
from mstar.engine.resources.step import StepContext
from mstar.model.glm5_next.components.causal_lm import Glm5NextForCausalLM
from mstar.model.glm5_next.config import KDA_STATE, KV_CACHE, SAMPLER, Glm5NextModelConfig
from mstar.model.glm5_next.glm5_next_model import Glm5NextModel, process_weights_after_loading
from mstar.model.glm5_next.submodules import Glm5NextLLMSubmodule
from mstar.model.submodule_base import InputMetadata
from mstar.worker.engine_manager import (
    _refuse_uncacheable_positions,
    _refuse_unknown_walks,
    _refuse_unskippable_resources,
)

PAGE = 16
WINDOW = 48  # prompts below run in several windows, each a snapshot point
MAX_TOKENS = 10


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


def _build_engine(k: int, tmp_path, kv_cache: bool = True) -> tuple[Engine, list[int]]:
    """The reduced long-context model with 16-token pages; returns the engine and a log of
    every prefix length its probes agreed. ``kv_cache`` False turns the KV prefix cache off,
    as a deployment's yaml does."""
    torch.manual_seed(0)
    cfg = Glm5NextModelConfig.reduced_long_context()
    cfg.mtp_num_draft_tokens = k
    cfg.prefill_window_tokens = WINDOW
    model = Glm5NextModel("x", config_variant="reduced", kda_conv_dtype=torch.float32,
                          kda_max_requests=4, mtp_num_draft_tokens=k)
    model.config = cfg
    model.checkpoint_path = lambda: str(tmp_path)  # names the weights, nothing to read
    lm = Glm5NextForCausalLM(cfg)
    torch.manual_seed(0)
    for p in lm.parameters():
        if p.dtype.is_floating_point:
            torch.nn.init.normal_(p, std=0.1)
    process_weights_after_loading(lm, torch.device("cpu"))
    lm.eval()
    specs = model.get_node_resources()
    for spec in specs:
        config = getattr(spec, "config", None)
        if isinstance(config, PagedKVConfig):
            config.page_size = PAGE
            config.prefix_cache = kv_cache
        if isinstance(config, RecurrentStateConfig):
            config.prefix_page_size = PAGE
            assert config.prefix_slots
    # what the worker refuses before it builds the engine
    _refuse_uncacheable_positions(specs, model)
    _refuse_unskippable_resources(specs, model)
    _refuse_unknown_walks(model)
    sub = Glm5NextLLMSubmodule(lm, cfg)
    engine = Engine(None)
    engine.load_model(
        {"LLM": sub}, specs,
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
    engine.warmup()
    hits: list[int] = []
    resolve = engine._runner.resolve_cached_prefix

    def logged(*args):
        prefix = resolve(*args)
        hits.append(0 if prefix is None else prefix.tokens)
        return prefix

    engine._runner.resolve_cached_prefix = logged
    return engine, hits


def _step(engine, walk, inputs, decode_iter=0):
    rids = list(inputs)
    infos = {rid: CurrentForwardPassInfo(
        request_id=f"wire-{rid}", rid_handle=rid, graph_walk=walk, fwd_index=0, random_seed=0,
        max_tokens=MAX_TOKENS,
        resource_configs={SAMPLER: SamplingReqConfig(temperature=0.0, ignore_eos=True)},
        dynamic_loop_iter_counts={"decode_loop": decode_iter}) for rid in rids}
    batch = ExecutingBatch(
        node_name="LLM", per_request_info=infos,
        per_request_input_tensors={rid: {"text_inputs": [ids]} for rid, ids in inputs.items()},
        per_request_input_metadata={rid: InputMetadata(
            dynamic_loop_iter_counts={"decode_loop": decode_iter}) for rid in rids},
        step_context=StepContext(request_ids=tuple(rids), graph_walk=walk, slot=0,
                                 capture=False),
    )
    engine.prepare_inputs(batch)
    assert not batch.failed_requests, batch.failed_requests
    outputs = engine.exec_and_postprocess(batch)
    assert batch.admit_error is None, batch.admit_error
    engine.finalize_batch(batch)
    stops = engine.check_stop_for_batch(batch, outputs)
    return outputs.per_rid_outputs, stops


def _kda_state(engine, rid: str) -> list[torch.Tensor]:
    """The request's recurrent state and conv window, every KDA layer."""
    pool = engine._resources[KDA_STATE]
    slot = pool._slots[rid]["main"].index
    return [pool.block(name, layer)[slot].clone() for name in ("state", "conv")
            for layer in range(pool.config.num_layers)]


def _configs(prompt: list[int], keyed: bool = True) -> dict:
    configs = {SAMPLER: SamplingReqConfig(temperature=0.0, ignore_eos=True)}
    if keyed:
        whole = len(prompt) // PAGE
        keys = {"main": chain([prompt[at:at + PAGE] for at in range(0, len(prompt), PAGE)])}
        tail = {"main": prompt[whole * PAGE:]}
        # the one ids span the preprocess worker lays a text prompt out as
        layout = {"main": [PrefixSpan(len(prompt), len(prompt), "prefill")]}
        for key in (KV_CACHE, KDA_STATE):
            configs[key] = KVReqConfig(prefix_keys=keys, prefix_tail=tail, prefix_layout=layout)
    return configs


def _generate(engine, rid: str, prompt: list[int], keyed: bool = True, decode: bool = True):
    engine.add_request(rid, _configs(prompt, keyed))
    out, stops = _step(engine, "prefill", {rid: torch.tensor(prompt)})
    state = _kda_state(engine, rid)
    seq = out[rid]["new_token"][0].tolist()
    nxt, step = out[rid]["text_inputs"][0], 0
    while decode and not stops.get(rid):
        out, stops = _step(engine, "decode", {rid: nxt}, step)
        step += 1
        seq += out[rid]["new_token"][0].tolist()
        nxt = out[rid]["text_inputs"][0]
    # requests hold their state until removed; keep "a"'s pages and snapshots indexed
    engine.remove_request(rid)
    return seq[:MAX_TOKENS], state


def _prompt(n: int, seed: int) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(2, 240, (n,), generator=g).tolist()


# k > 0 too once MTP serves prefix hits again (Glm5NextModel.prefix_key_streams)
@pytest.mark.parametrize("k", [0])
def test_a_prompt_past_a_cached_prefix_generates_what_a_fresh_prefill_does(k, tmp_path):
    a = _prompt(150, seed=1)
    cases = {
        # a's windows end at 48, 96 and 144, each a snapshot; b shares all 9 whole pages
        "b": (a[:144] + _prompt(30, seed=2), 144),
        # c shares 6 pages and a bit: the KV cache matches 96, and so does a snapshot
        "c": (a[:100] + _prompt(25, seed=3), 96),
    }
    engine, hits = _build_engine(k, tmp_path)
    _generate(engine, "a", a)
    for rid, (prompt, skipped) in cases.items():
        hits.clear()
        got, state = _generate(engine, rid, prompt)
        assert hits and hits[0] == skipped, (rid, hits)
        fresh, _ = _build_engine(k, tmp_path)
        want, want_state = _generate(fresh, rid, prompt, keyed=False)
        assert got == want, (rid, got, want)
        # the restored state carries on exactly where the whole prompt's would be: a small
        # random model's tokens barely move without it, so the state itself is compared
        for have, ref in zip(state, want_state, strict=True):
            assert torch.equal(have, ref), rid


@pytest.mark.parametrize("k", [0])
def test_prompts_prefilled_in_one_step_each_save_their_state(k, tmp_path):
    """a and o share a prefill step; each saves its state where its own pages end, so a next
    turn of either skips its first 144 tokens and lands where a fresh prefill does."""
    firsts = {"a": _prompt(150, seed=1), "o": _prompt(150, seed=7)}
    engine, hits = _build_engine(k, tmp_path)
    for rid, prompt in firsts.items():
        engine.add_request(rid, _configs(prompt))
    _step(engine, "prefill", {rid: torch.tensor(prompt) for rid, prompt in firsts.items()})
    for rid in firsts:
        engine.remove_request(rid)
    for rid, prompt in firsts.items():
        hits.clear()
        nxt = prompt[:144] + _prompt(30, seed=2)
        got, state = _generate(engine, f"{rid}2", nxt, decode=False)
        assert hits and hits[0] == 144, (rid, hits)
        want, want_state = _generate(_build_engine(k, tmp_path)[0], f"{rid}2", nxt, keyed=False,
                                     decode=False)
        assert got == want, rid
        # o's windows also end where a's do, so its state matches to rounding
        for have, ref in zip(state, want_state, strict=True):
            torch.testing.assert_close(have, ref, rtol=0, atol=1e-6, msg=rid)


def test_a_prefix_no_snapshot_reaches_is_served_whole(tmp_path):
    """The KV cache holds a's first pages, but a snapshot only at window ends: d shares 40
    tokens (2 pages), under a's first window, so nothing is skipped."""
    a = _prompt(150, seed=1)
    engine, hits = _build_engine(0, tmp_path)
    _generate(engine, "a", a)
    hits.clear()
    d = a[:40] + _prompt(60, seed=4)
    got, _ = _generate(engine, "d", d)
    assert hits and hits[0] == 0, hits
    fresh, _ = _build_engine(0, tmp_path)
    assert got == _generate(fresh, "d", d, keyed=False)[0]


def test_with_the_kv_cache_off_no_snapshot_is_taken_or_restored(tmp_path):
    """kv_cache.prefix_cache false closes the KDA snapshots too: a snapshot alone would skip
    tokens whose MLA rows are not cached. b runs whole, as a fresh prefill does."""
    a = _prompt(150, seed=1)
    b = a[:144] + _prompt(10, seed=2)
    engine, hits = _build_engine(0, tmp_path, kv_cache=False)
    assert not engine._resources[KDA_STATE].keeps_snapshots
    _generate(engine, "a", a)
    got, _ = _generate(engine, "b", b)
    assert not hits, hits
    assert not engine._resources[KDA_STATE]._snap_by_key
    fresh, _ = _build_engine(0, tmp_path)
    assert got == _generate(fresh, "b", b, keyed=False)[0]


def test_a_long_prompt_snapshots_its_last_page_first():
    """More windows than a prompt may snapshot: the last whole page, which a next turn
    extending the prompt needs, then the newest window ends."""
    from mstar.model.glm5_next.submodules import _SNAPSHOTS_PER_PROMPT

    sub = types.SimpleNamespace(
        _snapshot_page=PAGE, request_states={},
        config=types.SimpleNamespace(prefill_window_tokens=WINDOW, dsa_long_context=True))
    n = 10 * WINDOW + PAGE + 3
    kept, spares = Glm5NextLLMSubmodule._snapshot_points(
        sub, "prefill", ["r"], [types.SimpleNamespace(input_seq_len=n)], None)
    newest = [n // PAGE * PAGE] + [w * WINDOW for w in range(10, 0, -1)]
    assert [p for _, _, p in kept] == newest[:1]
    assert [p for _, _, p in spares] == newest[1:_SNAPSHOTS_PER_PROMPT]


def test_more_sessions_than_prompts_fill_the_pool_keep_their_next_turn(tmp_path):
    """Each first turn reserves its last page and three window ends: four fill the 16
    snapshot slots. A later session's last page takes a window end's slot, so every
    session's next turn still skips its whole first turn."""
    engine, hits = _build_engine(0, tmp_path)
    assert engine._resources[KDA_STATE].config.prefix_slots == 16
    firsts = [_prompt(250, seed=10 + s) for s in range(8)]
    for s, prompt in enumerate(firsts):
        _generate(engine, f"s{s}-t1", prompt, decode=False)
    for s, prompt in enumerate(firsts):
        hits.clear()
        _generate(engine, f"s{s}-t2", prompt[:240] + _prompt(20, seed=99), decode=False)
        assert hits[0] == 240, (s, hits)


def test_a_refused_admits_skipped_prefix_does_not_outlive_it(tmp_path):
    """A hit's split marks its prefix skipped; a retry that no longer hits plans whole."""
    engine, _ = _build_engine(0, tmp_path)
    engine.add_request("e", {SAMPLER: SamplingReqConfig(temperature=0.0, ignore_eos=True)})
    engine._submodules["LLM"].submodule.request_state("e").add("cached_prefix", 96)
    out, _ = _step(engine, "prefill", {"e": torch.tensor(_prompt(150, seed=5))})
    assert out["e"]["new_token"][0].numel() == 1
    engine.remove_request("e")
