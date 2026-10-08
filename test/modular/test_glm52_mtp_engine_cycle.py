"""GLM-5.2 MTP on the resource-pools engine, CPU, reduced config."""
from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _cpu_rmsnorm(x, weight, eps=1e-6):
    x32 = x.float()
    normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return (normed * weight.float()).to(x.dtype)


def _cpu_flashinfer() -> types.ModuleType:
    fi = types.ModuleType("flashinfer")
    fi.norm = types.SimpleNamespace(rmsnorm=_cpu_rmsnorm)
    return fi


try:
    import flashinfer  # noqa: F401  (the real one stays for later test files)
except ImportError:
    sys.modules["flashinfer"] = _cpu_flashinfer()


@pytest.fixture(autouse=True)
def _force_cpu_flashinfer(monkeypatch):
    # forced per test: on a box with real flashinfer an earlier import would
    # otherwise route CPU tensors into GPU kernels
    monkeypatch.setitem(sys.modules, "flashinfer", _cpu_flashinfer())


from mstar.engine.resources import AllocationFailed, StepContext  # noqa: E402
from mstar.model.glm52._testing import build_cpu_resources  # noqa: E402
from mstar.model.glm52.components.causal_lm import Glm52ForCausalLM  # noqa: E402
from mstar.model.glm52.config import (  # noqa: E402
    KV_RESOURCE,
    SAMPLER_RESOURCE,
    Glm52ModelConfig,
)
from mstar.model.glm52.quantization import process_weights_after_loading  # noqa: E402
from mstar.model.glm52.submodules import Glm52LLMSubmodule  # noqa: E402
from mstar.model.submodule_base import ModelInputsFromEngine  # noqa: E402

# small enough that two requests run it out mid-decode, big enough for one
POOL_PAGES = 10


def _cfg(k: int, mla_absorb: bool) -> Glm52ModelConfig:
    cfg = Glm52ModelConfig.reduced()
    cfg.num_hidden_layers = 4  # the MTP position lands FULL (4 = offset-1 + freq)
    cfg.mtp_num_draft_tokens = k
    cfg.mla_absorb = mla_absorb
    return cfg


def _fwd_info(rid, max_tokens: int, ignore_eos: bool, wire: str | None = None) -> SimpleNamespace:
    # the worker keys a request by its handle; request_id is the wire id
    return SimpleNamespace(
        request_id=rid if wire is None else wire,
        rid_handle=rid,
        max_tokens=max_tokens,
        resource_configs={SAMPLER_RESOURCE: SimpleNamespace(
            ignore_eos=ignore_eos, temperature=0.0, repetition_penalty=1,
        )},
        dynamic_loop_iter_counts={},
    )


class _Driver:
    """The engine's per-step cycle for one node, over real resources: declare, admit,
    plan, preprocess, forward, commit, the host's cut of packed outputs, postprocess."""

    def __init__(self, sub: Glm52LLMSubmodule, cfg: Glm52ModelConfig, rids: list[str], **pool):
        self.sub = sub
        self.resources, self.runner = build_cpu_resources(cfg, rids, **pool)
        sub.bind_node_resources(self.resources)

    def step(self, walk: str, batch: dict[str, tuple[SimpleNamespace, torch.Tensor]],
             refusable: bool = False):
        """``refusable``: a refused admit comes back as its outcome, as the
        engine hands it to the worker, and the forward never runs."""
        rids = list(batch)
        for info, _ in batch.values():
            info.graph_walk = walk
        inputs = [
            self.sub.prepare_inputs(walk, info, {"text_inputs": [text]})
            for info, text in batch.values()
        ]
        step = self.sub.declare_step(walk, rids, inputs)
        ctx = StepContext(request_ids=tuple(rids), graph_walk=walk, slot=0, capture=False)
        step.set_ctx(ctx)
        outcome = self.runner.admit(step)
        if refusable and not outcome.ok:
            return outcome
        assert outcome.ok
        self.runner.plan(step)
        infos = {rid: info for rid, (info, _) in batch.items()}
        engine_inputs = ModelInputsFromEngine(
            request_ids=rids, per_request_info=infos, resources=self.resources, step=step,
        )
        kw = self.sub.preprocess(walk, engine_inputs, inputs)
        outs = self.sub.forward_batched(walk, engine_inputs, **kw)
        self.runner.commit(step)
        unpacked = self.sub.unpack_packed_outputs(
            static_output=outs, request_ids=rids,
            real_seq_lens=[inp.input_seq_len for inp in inputs], inputs=inputs,
            per_request_info=infos,
        )
        for rid, rid_out in unpacked.items():
            outs[rid].update(rid_out)
        for rid, (info, _) in batch.items():
            self.sub.postprocess(rid, info, outs[rid])
        return outs


def _drive(driver: _Driver, prompt: torch.Tensor, info, max_steps=64) -> torch.Tensor:
    rid = info.rid_handle
    emitted = []
    walk, text, decode_step = "prefill", prompt, 0
    for _ in range(max_steps):
        out = driver.step(walk, {rid: (info, text)})[rid]
        emitted.append(out["new_token"][0])
        if walk == "decode":
            info.dynamic_loop_iter_counts["decode_loop"] = decode_step
            decode_step += 1
        if driver.sub.check_stop(rid, info, out):
            break
        walk, text = "decode", out["text_inputs"][0]
    return torch.cat(emitted)


def _model(cfg: Glm52ModelConfig, seed: int = 0) -> Glm52ForCausalLM:
    """Every parameter randomized: the MoE expert containers are raw
    ``torch.empty`` at construction (the loader fills them), and garbage
    there NaNs the logits — a NaN model emits an all-zero argmax stream on
    every path, which makes a stream comparison vacuous.
    """
    torch.manual_seed(seed)
    model = Glm52ForCausalLM(cfg)
    for name, p in model.named_parameters():
        if "norm" in name:
            p.data.normal_(1.0, 0.02)
        elif name.endswith("gate.weight") or "e_score_correction_bias" in name:
            p.data.normal_(0, 1.0)
        else:
            p.data.normal_(0, 0.05)
    process_weights_after_loading(model, torch.device("cpu"))
    return model.eval()


def _run_pair(k, max_tokens, ignore_eos, mla_absorb, seed=0, eos_ids=None, oracle=None):
    """One model, two runs: MTP off, then on. Same weights by construction. ``oracle``:
    the MTP run drafts from the plain run's stream, wrong at every ``oracle``-th draft
    (0: never)."""
    cfg = _cfg(k, mla_absorb)
    if eos_ids is not None:
        cfg.eos_token_ids = eos_ids
    model = _model(cfg, seed)
    prompt = torch.arange(5, dtype=torch.long) + 3
    streams, drivers = [], []
    for mode_k in (0, k):
        cfg.mtp_num_draft_tokens = mode_k
        sub = Glm52LLMSubmodule(model, cfg)
        if mode_k and oracle is not None:
            _oracle(sub, {"r0": streams[0].tolist()}, oracle)
        driver = _Driver(sub, cfg, ["r0"])
        streams.append(_drive(driver, prompt, _fwd_info("r0", max_tokens, ignore_eos)))
        drivers.append(driver)
    cfg.mtp_num_draft_tokens = k
    return streams, drivers


def _oracle(sub: Glm52LLMSubmodule, reference: dict, wrong_every: int):
    """Drafts from the reference continuation at each request's position, wrong at every
    ``wrong_every``-th call when nonzero. A step drafts d2..dk in its draft loop, then the
    next step's d1 once its verdict is in (the prefill: the first d1); wrapping the two
    passes tells the oracle which call is which and how far each request is."""
    calls = {"n": 0}
    where: dict = {}
    given: dict = {}  # each request's drafts for its current step, d1 first
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
        where.update(rids=rids, row=0, done={rid: sub._mtp_emitted[rid] for rid in rids})
        return real_decode(engine_inputs, *args)

    def draft(h_head, prev):
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


def test_mtp_keys_request_state_by_handle():
    cfg = _cfg(2, True)
    model = _model(cfg)
    prompt = torch.arange(5, dtype=torch.long) + 3
    streams = []
    for wire in (None, "req-a"):
        sub = Glm52LLMSubmodule(model, cfg)
        streams.append(_drive(_Driver(sub, cfg, [7]), prompt, _fwd_info(7, 14, True, wire)))
        sub.cleanup_request(7)
        assert not sub._mtp_max_tokens and not sub._mtp_ignore_eos
    assert torch.equal(streams[0], streams[1])


@pytest.mark.parametrize("mla_absorb", [True, False], ids=["absorbed", "naive"])
def test_mtp_stream_matches_baseline_bitwise(mla_absorb):
    (base, spec), _ = _run_pair(k=2, max_tokens=24, ignore_eos=True, mla_absorb=mla_absorb)
    assert torch.equal(base, spec), f"{base.tolist()} vs {spec.tolist()}"
    assert base.numel() == 24
    # a real stream, not a degenerate one (an uninitialized model emits 0s)
    assert torch.isfinite(base.float()).all() and len(set(base.tolist())) > 3


@pytest.mark.parametrize("mla_absorb", [True, False], ids=["absorbed", "naive"])
def test_mtp_stream_matches_with_k3(mla_absorb):
    (base, spec), _ = _run_pair(k=3, max_tokens=20, ignore_eos=True, mla_absorb=mla_absorb)
    assert torch.equal(base, spec)


@pytest.mark.parametrize("k", [1, 3])
@pytest.mark.parametrize("wrong_every", [0, 3], ids=["oracle", "oracle_partial"])
def test_accepted_drafts_keep_the_stream(k, wrong_every):
    """Drafts that the verify accepts (all of them, or all but every third) emit the
    plain stream: random weights reject nearly every MTP draft, so only an oracle reaches
    the accepted rows, the KV trim at the next declare and the seed at the accepted row."""
    (base, spec), _ = _run_pair(k=k, max_tokens=20, ignore_eos=True, mla_absorb=True,
                                oracle=wrong_every)
    assert torch.equal(base, spec), f"{base.tolist()} vs {spec.tolist()}"


def test_oracle_drafts_are_all_accepted():
    """With always-right drafts every decode step emits k + 1 tokens."""
    k = 3
    (base, _), _ = _run_pair(k=k, max_tokens=16, ignore_eos=True, mla_absorb=True)
    cfg = _cfg(k, True)
    sub = Glm52LLMSubmodule(_model(cfg), cfg)
    _oracle(sub, {"r0": base.tolist()}, 0)
    driver = _Driver(sub, cfg, ["r0"])
    info = _fwd_info("r0", 16, True)
    out = driver.step("prefill", {"r0": (info, torch.arange(5, dtype=torch.long) + 3)})["r0"]
    got = out["new_token"][0].tolist()
    for _ in range(2):
        out = driver.step("decode", {"r0": (info, out["text_inputs"][0])})["r0"]
        emitted = out["new_token"][0].tolist()
        assert len(emitted) == k + 1
        got += emitted
        assert got == base.tolist()[:len(got)]


def test_mtp_stops_exactly_at_max_tokens():
    for k in (1, 2, 3):
        (base, spec), _ = _run_pair(k=k, max_tokens=11, ignore_eos=True, mla_absorb=True)
        assert base.numel() == 11 and torch.equal(base, spec)


def test_mtp_eos_truncation_matches_baseline():
    # make an eos id likely: the greedy stream over random weights repeats
    cfg = _cfg(2, True)
    model = _model(cfg)
    prompt = torch.arange(5, dtype=torch.long) + 3
    cfg.mtp_num_draft_tokens = 0
    sub = Glm52LLMSubmodule(model, cfg)
    base = _drive(_Driver(sub, cfg, ["r0"]), prompt, _fwd_info("r0", 12, True))
    eos = int(base[6])
    cfg.eos_token_ids = (eos,)
    streams = []
    for mode_k in (0, 2):
        cfg.mtp_num_draft_tokens = mode_k
        sub = Glm52LLMSubmodule(model, cfg)
        streams.append(_drive(_Driver(sub, cfg, ["r0"]), prompt, _fwd_info("r0", 12, False)))
    assert torch.equal(streams[0], streams[1])
    assert int(streams[1][-1]) == eos and streams[1].numel() <= 7


@pytest.mark.parametrize("wrong_every", [None, 0], ids=["mtp", "oracle"])
def test_kv_length_tracks_the_verified_stream(wrong_every):
    """Every step commits its whole block; the next declare takes back the rows the verdict
    rejected, so each step starts from the prompt plus the emitted tokens but the last."""
    k = 2
    (base, _), _ = _run_pair(k=k, max_tokens=20, ignore_eos=True, mla_absorb=True)
    cfg = _cfg(k, True)
    sub = Glm52LLMSubmodule(_model(cfg), cfg)
    if wrong_every is not None:
        _oracle(sub, {"r0": base.tolist()}, wrong_every)
    driver = _Driver(sub, cfg, ["r0"])
    info = _fwd_info("r0", 20, True)
    kv = driver.resources[KV_RESOURCE]
    prompt = torch.arange(5, dtype=torch.long) + 3
    out = driver.step("prefill", {"r0": (info, prompt)})["r0"]
    total = prompt.numel() + 1
    assert kv.stored_len("r0") == prompt.numel()
    for _ in range(4):
        inputs = [sub.prepare_inputs("decode", info, {"text_inputs": out["text_inputs"]})]
        sub.declare_step("decode", ["r0"], inputs)  # the trim; idempotent at the real declare
        assert kv.stored_len("r0") == total - 1
        out = driver.step("decode", {"r0": (info, out["text_inputs"][0])})["r0"]
        _, _, accepted = sub._mtp_verdict["r0"]
        assert out["new_token"][0].numel() == accepted + 1
        assert kv.stored_len("r0") == total - 1 + k + 1
        assert out["text_inputs"][0].numel() == 1
        total += accepted + 1


def test_batch_of_two_matches_single_streams():
    """Two requests in one batch (different prompts, different accepted
    counts per step) emit what each emits alone."""
    cfg = _cfg(2, True)
    model = _model(cfg)
    prompts = {"a": torch.arange(5, dtype=torch.long) + 3, "b": torch.arange(3, dtype=torch.long) + 9}
    singles = {}
    for rid, prompt in prompts.items():
        sub = Glm52LLMSubmodule(model, cfg)
        singles[rid] = _drive(_Driver(sub, cfg, [rid]), prompt, _fwd_info(rid, 14, True))

    sub = Glm52LLMSubmodule(model, cfg)
    driver = _Driver(sub, cfg, list(prompts))
    infos = {rid: _fwd_info(rid, 14, True) for rid in prompts}
    emitted = {rid: [] for rid in prompts}
    texts = {}
    for rid, prompt in prompts.items():
        out = driver.step("prefill", {rid: (infos[rid], prompt)})[rid]
        emitted[rid].append(out["new_token"][0])
        assert not sub.check_stop(rid, infos[rid], out)
        texts[rid] = out["text_inputs"][0]
    live = set(prompts)
    for _ in range(20):
        if not live:
            break
        outs = driver.step("decode", {rid: (infos[rid], texts[rid]) for rid in sorted(live)})
        for rid in sorted(live):
            emitted[rid].append(outs[rid]["new_token"][0])
            texts[rid] = outs[rid]["text_inputs"][0]
            if sub.check_stop(rid, infos[rid], outs[rid]):
                live.discard(rid)
    for rid in prompts:
        got = torch.cat(emitted[rid])
        assert torch.equal(got, singles[rid]), f"{rid}: {got.tolist()} vs {singles[rid].tolist()}"


def test_non_greedy_request_is_refused_under_mtp():
    cfg = _cfg(2, True)
    sub = Glm52LLMSubmodule(_model(cfg), cfg)
    info = _fwd_info("r0", 8, True)
    info.resource_configs[SAMPLER_RESOURCE].temperature = 0.7
    with pytest.raises(RuntimeError, match="greedy-only"):
        sub.prepare_inputs("prefill", info, {"text_inputs": [torch.tensor([1, 2, 3])]})


def test_declare_step_shapes():
    cfg = _cfg(0, True)
    sub = Glm52LLMSubmodule(_model(cfg), cfg)
    prompt = {"text_inputs": [torch.tensor([1, 2, 3])]}
    inputs = [sub.prepare_inputs("prefill", _fwd_info("r0", 8, True), prompt)]
    step = sub.declare_step("prefill", ["r0"], inputs)
    assert set(step.keys()) == {"kv", "attn", "sampler"}
    assert step.segments[0].span == 3
    cfg.mtp_num_draft_tokens = 2
    sub = Glm52LLMSubmodule(_model(cfg), cfg)
    sub.bind_node_resources(build_cpu_resources(cfg, ["r0"])[0])
    inputs = [sub.prepare_inputs("prefill", _fwd_info("r0", 8, True), prompt)]
    step = sub.declare_step("prefill", ["r0"], inputs)
    assert set(step.keys()) == {"kv", "attn", "sampler"}
    assert step.segments[0].span == 3
    # one id in, the block of k+1 rows planned; the verify and drafts are argmaxes
    inputs = [sub.prepare_inputs("decode", _fwd_info("r0", 8, True),
                                 {"text_inputs": [torch.tensor([4])]})]
    step = sub.declare_step("decode", ["r0"], inputs)
    assert set(step.keys()) == {"kv", "attn"}
    assert step.segments[0].span == 3


def test_seed_slots_are_reused():
    cfg = _cfg(2, True)
    model = _model(cfg)
    sub = Glm52LLMSubmodule(model, cfg)
    driver = _Driver(sub, cfg, ["a", "b"])
    free = len(sub._mtp_free_slots)
    prompt = torch.arange(5, dtype=torch.long) + 3
    _drive(driver, prompt, _fwd_info("a", 6, True))
    assert sub._mtp_slot["a"] != 0 and len(sub._mtp_free_slots) == free - 1
    sub.cleanup_request("a")
    assert len(sub._mtp_free_slots) == free and not sub._mtp_pending


# ── the context limit ────────────────────────────────────────────────────


def _window_cfg(k: int, limit: int) -> Glm52ModelConfig:
    """Reduced config whose context limit (index_topk, DSA off) is ``limit``
    rows: two CPU pages of 8, so a row past the limit needs a third page."""
    cfg = _cfg(k, mla_absorb=True)
    cfg.index_topk = limit
    return cfg


def _window_stream(model, cfg, mode_k: int, prompt_len: int):
    """One arm at the window: the stream, the capped budget, and the KV
    stream's final length and page high-water mark."""
    k = cfg.mtp_num_draft_tokens
    cfg.mtp_num_draft_tokens = mode_k
    try:
        sub = Glm52LLMSubmodule(model, cfg)
        driver = _Driver(sub, cfg, ["r0"])
        prompt = torch.arange(prompt_len, dtype=torch.long) + 3
        stream = _drive(driver, prompt, _fwd_info("r0", 40, True))
        kv = driver.resources[KV_RESOURCE]
        pages = len(kv._streams["r0"]["main"].page_indices)
        return stream, sub._token_budget["r0"], kv.stored_len("r0"), pages
    finally:
        cfg.mtp_num_draft_tokens = k


@pytest.mark.parametrize("k", [2, 3])
def test_stream_stops_cleanly_at_the_context_limit(k):
    """A request whose max_tokens outruns the context limit stops short of it: the
    prefill's preprocess caps its budget at limit - prompt, less two blocks under MTP
    (every step writes a whole block, and the next may run before the stop lands), so the
    trunk guard never fires and no step writes past the limit (no third KV page)."""
    limit, prompt_len = 24, 5
    cfg = _window_cfg(k, limit)
    model = _model(cfg)
    baseline, budget, stored, pages = _window_stream(model, cfg, 0, prompt_len)
    assert budget == limit - prompt_len
    assert baseline.numel() == budget
    assert stored == limit - 1 and pages == limit // 8
    stream, budget, stored, pages = _window_stream(model, cfg, k, prompt_len)
    assert budget == limit - prompt_len - 2 * (k + 1)
    assert torch.equal(stream, baseline[:budget]), (stream.tolist(), baseline.tolist())
    assert stored <= limit and pages <= limit // 8, (stored, pages)


def test_budget_within_the_window_is_untouched():
    cfg = _window_cfg(2, 64)
    model = _model(cfg)
    sub = Glm52LLMSubmodule(model, cfg)
    driver = _Driver(sub, cfg, ["r0"])
    prompt = torch.arange(5, dtype=torch.long) + 3
    stream = _drive(driver, prompt, _fwd_info("r0", 9, True))
    assert sub._token_budget["r0"] == 9
    assert stream.numel() == 9


# ── KV exhaustion ────────────────────────────────────────────────────────


def test_the_forward_allocates_no_pages():
    """Every page an MTP step writes (the trunk's block and the MTP plane's, on one page
    table) is admitted with the step, so running out refuses the step instead of raising
    inside forward_batched and failing the whole batch. With the arena refusing every
    allocation inside the forward, the stream is unchanged."""
    k = 2
    (base, _), _ = _run_pair(k=k, max_tokens=16, ignore_eos=True, mla_absorb=True)
    cfg = _cfg(k, True)
    sub = Glm52LLMSubmodule(_model(cfg), cfg)
    driver = _Driver(sub, cfg, ["r0"])
    arena = driver.resources[KV_RESOURCE]._arena
    forward = sub.forward_batched

    def no_pages_inside(*args, **kwargs):
        acquire = arena.acquire
        arena.acquire = lambda n: None
        try:
            return forward(*args, **kwargs)
        finally:
            arena.acquire = acquire

    sub.forward_batched = no_pages_inside
    spec = _drive(driver, torch.arange(5, dtype=torch.long) + 3, _fwd_info("r0", 16, True))
    assert torch.equal(base, spec)


def test_exhaustion_refuses_the_step_and_the_pool_recovers():
    """merceod's small-pool run on CPU. The refusal comes from the step's
    admission as ``AllocationFailed``, which the worker re-queues, not from
    inside the forward, and nothing the refused step would have written
    moves. Here the refused batch's second request fails and is removed: the
    first completes as it would alone, the pool ends where it started, and a
    new request on it completes too."""
    k, max_tokens = 2, 14
    cfg = _cfg(k, True)
    model = _model(cfg)
    prompts = {"a": torch.arange(5, dtype=torch.long) + 3, "b": torch.arange(3, dtype=torch.long) + 9}
    singles = {
        rid: _drive(_Driver(Glm52LLMSubmodule(model, cfg), cfg, [rid]), p,
                    _fwd_info(rid, max_tokens, True))
        for rid, p in prompts.items()
    }

    sub = Glm52LLMSubmodule(model, cfg)
    driver = _Driver(sub, cfg, list(prompts), max_num_pages=POOL_PAGES, page_size=4)
    kv = driver.resources[KV_RESOURCE]
    free = kv._arena.num_free
    infos = {rid: _fwd_info(rid, max_tokens, True) for rid in prompts}
    emitted = {rid: [] for rid in prompts}
    texts = {}

    def remove(rid):
        driver.runner.remove_request(rid)
        sub.cleanup_request(rid)

    for rid, prompt in prompts.items():
        out = driver.step("prefill", {rid: (infos[rid], prompt)})[rid]
        emitted[rid].append(out["new_token"][0])
        assert not sub.check_stop(rid, infos[rid], out)
        texts[rid] = out["text_inputs"][0]
    live, refused = set(prompts), 0
    for _ in range(64):
        if not live:
            break
        # the last verdict's trim lands at declare, refused or not, and only once
        for rid in live:
            sub._apply_verdict(rid)
        before = {rid: kv.stored_len(rid) for rid in live}
        outs = driver.step("decode", {rid: (infos[rid], texts[rid]) for rid in sorted(live)},
                           refusable=True)
        if not isinstance(outs, dict):
            assert isinstance(outs.reason, AllocationFailed), outs
            assert {rid: kv.stored_len(rid) for rid in live} == before
            assert live == set(prompts)
            refused += 1
            live.discard("b")
            remove("b")
            continue
        for rid in sorted(live):
            emitted[rid].append(outs[rid]["new_token"][0])
            texts[rid] = outs[rid]["text_inputs"][0]
            if sub.check_stop(rid, infos[rid], outs[rid]):
                live.discard(rid)
                remove(rid)
    assert refused == 1
    assert torch.equal(torch.cat(emitted["a"]), singles["a"])
    assert kv._arena.num_free == free

    driver.runner.ingest_request("c")
    again = _drive(driver, prompts["b"], _fwd_info("c", max_tokens, True))
    assert torch.equal(again, singles["b"])
    remove("c")
    assert kv._arena.num_free == free


def test_stop_counts_what_the_checked_steps_delivered():
    # async scheduling runs step N+1 before check_stop(N); the stop must count through N
    cfg = _cfg(3, True)
    model = _model(cfg)
    for budget in range(4, 12):
        sub = Glm52LLMSubmodule(model, cfg)
        driver = _Driver(sub, cfg, [7])
        info = _fwd_info(7, budget, True)
        out = driver.step("prefill", {7: (info, torch.arange(5, dtype=torch.long) + 3)})[7]
        delivered = out["new_token"][0].numel()
        assert not sub.check_stop(7, info, out)
        cur = driver.step("decode", {7: (info, out["text_inputs"][0])})[7]
        for _ in range(40):
            nxt = driver.step("decode", {7: (info, cur["text_inputs"][0])})[7]
            delivered += cur["new_token"][0].numel()
            if sub.check_stop(7, info, cur):
                break
            cur = nxt
        assert delivered == budget, (budget, delivered)


@pytest.mark.parametrize("wrong_every", [None, 0, 3], ids=["mtp", "oracle", "oracle_partial"])
@pytest.mark.parametrize("chunk", [8192, 3], ids=["one-pass", "row-chunks"])
def test_mtp_stream_matches_baseline_past_topk_on_paged_dsa(chunk, wrong_every):
    """MTP on paged DSA, every context past index_topk: the verify and the seed pass select
    per row, a rejected row's index keys are trimmed with its latents, and the emitted
    stream is plain decode's. Row chunks: the prompt's trunk and MTP passes run in pieces.
    Random weights reject nearly every draft; the oracle reaches the accepted rows."""
    cfg = _cfg(2, True)
    cfg.dsa_long_context, cfg.index_topk = True, 4
    cfg.prefill_chunk_tokens = chunk
    model = _model(cfg)
    prompt = torch.tensor([5, 9, 2, 7, 1, 8, 3])
    streams = []
    for k in (0, 2):
        cfg.mtp_num_draft_tokens = k
        sub = Glm52LLMSubmodule(model, cfg)
        if k and wrong_every is not None:
            _oracle(sub, {"r0": streams[0].tolist()}, wrong_every)
        driver = _Driver(sub, cfg, ["r0"])
        streams.append(_drive(driver, prompt, _fwd_info("r0", 18, True)))
        index = driver.resources["kv_index"]
        assert index.stored_len("r0") == driver.resources[KV_RESOURCE].stored_len("r0")
    cfg.mtp_num_draft_tokens = 2
    assert torch.equal(streams[0], streams[1]), f"{streams[0].tolist()} vs {streams[1].tolist()}"
    assert len(set(streams[0].tolist())) > 3
