"""GLM-5.2 on the resource-pools engine, CPU, reduced config."""
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


if "flashinfer" not in sys.modules:
    sys.modules["flashinfer"] = _cpu_flashinfer()


@pytest.fixture(autouse=True)
def _force_cpu_flashinfer(monkeypatch):
    # forced per test: on a box with real flashinfer an earlier import would
    # otherwise route CPU tensors into GPU kernels
    monkeypatch.setitem(sys.modules, "flashinfer", _cpu_flashinfer())


from mstar.engine.resources import StepContext  # noqa: E402
from mstar.model.glm52._testing import (  # noqa: E402
    EagerPiecewiseRunner,
    build_cpu_resources,
)
from mstar.model.glm52.components.causal_lm import Glm52ForCausalLM  # noqa: E402
from mstar.model.glm52.config import (  # noqa: E402
    KV_RESOURCE,
    SAMPLER_RESOURCE,
    Glm52ModelConfig,
)
from mstar.model.glm52.quantization import process_weights_after_loading  # noqa: E402
from mstar.model.glm52.submodules import (  # noqa: E402
    MTP_DRAFT_BUNDLE,
    Glm52LLMSubmodule,
)
from mstar.model.submodule_base import ModelInputsFromEngine  # noqa: E402


def _cfg(k: int, mla_absorb: bool) -> Glm52ModelConfig:
    cfg = Glm52ModelConfig.reduced()
    cfg.num_hidden_layers = 4  # the MTP position lands FULL (4 = offset-1 + freq)
    cfg.mtp_num_draft_tokens = k
    cfg.mla_absorb = mla_absorb
    return cfg


def _fwd_info(rid: str, max_tokens: int, ignore_eos: bool) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=rid,
        max_tokens=max_tokens,
        resource_configs={SAMPLER_RESOURCE: SimpleNamespace(
            ignore_eos=ignore_eos, temperature=0.0, repetition_penalty=1,
        )},
        dynamic_loop_iter_counts={},
    )


class _Driver:
    """The engine's per-step cycle for one node, over real resources."""

    def __init__(self, sub: Glm52LLMSubmodule, cfg: Glm52ModelConfig, rids: list[str],
                 regions: bool = False, batch_sizes=(1, 2, 4)):
        self.sub = sub
        self.resources, self.runner = build_cpu_resources(cfg, rids)
        sub.bind_node_resources(self.resources)
        self.piecewise = {}
        if regions:
            configs = sub.get_piecewise_cuda_graph_configs(
                torch.device("cpu"), torch.float32, tp_world_size=1,
            )
            self.piecewise = {
                label: EagerPiecewiseRunner(label, config, self.resources, self.runner, batch_sizes)
                for label, config in configs.items()
            }

    def step(self, walk: str, batch: dict[str, tuple[SimpleNamespace, torch.Tensor]]):
        rids = list(batch)
        inputs = [
            self.sub.prepare_inputs(walk, info, {"text_inputs": [text]})
            for info, text in batch.values()
        ]
        step = self.sub.declare_step(walk, rids, inputs)
        ctx = StepContext(request_ids=tuple(rids), graph_walk=walk, slot=0, capture=False)
        if step is not None:
            step.set_ctx(ctx)
            assert self.runner.admit(step).ok
            self.runner.plan(step)
        engine_inputs = ModelInputsFromEngine(
            request_ids=rids,
            per_request_info={rid: info for rid, (info, _) in batch.items()},
            resources=self.resources,
            piecewise_runners=self.piecewise,
            step=step,
        )
        kw = self.sub.preprocess(walk, engine_inputs, inputs)
        outs = self.sub.forward_batched(walk, engine_inputs, **kw)
        if step is not None:
            self.runner.commit(step)
        for rid, (info, _) in batch.items():
            self.sub.postprocess(rid, info, outs[rid])
        return outs


def _drive(driver: _Driver, prompt: torch.Tensor, info, max_steps=64,
           carry_prefill_drafts=False) -> torch.Tensor:
    rid = info.request_id
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
        nxt = out["text_inputs"][0]
        if carry_prefill_drafts and MTP_DRAFT_BUNDLE in out:
            nxt = out[MTP_DRAFT_BUNDLE][0]
        walk, text = "decode", nxt
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


def _run_pair(k, max_tokens, ignore_eos, mla_absorb, seed=0, eos_ids=None,
              carry_prefill_drafts=False, regions=False):
    """One model, two runs: MTP off, then on. Same weights by construction."""
    cfg = _cfg(k, mla_absorb)
    if eos_ids is not None:
        cfg.eos_token_ids = eos_ids
    model = _model(cfg, seed)
    prompt = torch.arange(5, dtype=torch.long) + 3
    streams, drivers = [], []
    for mode_k in (0, k):
        cfg.mtp_num_draft_tokens = mode_k
        sub = Glm52LLMSubmodule(model, cfg)
        driver = _Driver(sub, cfg, ["r0"], regions=regions and mode_k > 0)
        streams.append(_drive(
            driver, prompt, _fwd_info("r0", max_tokens, ignore_eos),
            carry_prefill_drafts=carry_prefill_drafts,
        ))
        drivers.append(driver)
    cfg.mtp_num_draft_tokens = k
    return streams, drivers


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


def test_prefill_drafts_carried_or_not_same_stream():
    (base, spec), _ = _run_pair(
        k=2, max_tokens=16, ignore_eos=True, mla_absorb=True, carry_prefill_drafts=True,
    )
    assert torch.equal(base, spec)


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


def test_kv_length_tracks_the_verified_stream():
    """After every MTP step the stream holds exactly the emitted tokens plus
    the prompt: the trunk's k+1 rows were committed and k+1-e rewound, the
    plane's transient rows never counted."""
    cfg = _cfg(2, True)
    model = _model(cfg)
    prompt = torch.arange(5, dtype=torch.long) + 3
    sub = Glm52LLMSubmodule(model, cfg)
    driver = _Driver(sub, cfg, ["r0"])
    info = _fwd_info("r0", 20, True)
    kv = driver.resources[KV_RESOURCE]
    out = driver.step("prefill", {"r0": (info, prompt)})["r0"]
    total = prompt.numel()
    assert kv.stored_len("r0") == total
    text = out["text_inputs"][0]
    for _ in range(4):
        out = driver.step("decode", {"r0": (info, text)})["r0"]
        e = out["new_token"][0].numel()
        total += e
        assert kv.stored_len("r0") == total
        assert out["text_inputs"][0].numel() == cfg.mtp_num_draft_tokens + 1
        text = out["text_inputs"][0]


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
    inputs = [sub.prepare_inputs("prefill", _fwd_info("r0", 8, True), {"text_inputs": [torch.tensor([1, 2, 3])]})]
    step = sub.declare_step("prefill", ["r0"], inputs)
    assert set(step.keys()) == {"kv", "attn", "sampler"}
    assert step.segments[0].span == 3
    cfg.mtp_num_draft_tokens = 2
    sub = Glm52LLMSubmodule(_model(cfg), cfg)
    assert set(sub.declare_step("prefill", ["r0"], inputs).keys()) == {"sampler"}
    assert sub.declare_step("decode", ["r0"], inputs) is None
    assert sub.get_cuda_graph_configs(torch.device("cpu")) == []


@pytest.mark.parametrize("k", [1, 2, 3])
def test_mtp_regions_match_baseline_bitwise(k):
    """The captured regions' code (trunk verify, one-graph draft phase with its k
    sub-plans, prefill trunk) run eagerly through the region contract emit the same
    stream as plain decode.
    """
    (base, spec), drivers = _run_pair(
        k=k, max_tokens=18, ignore_eos=True, mla_absorb=True, regions=True,
    )
    assert torch.equal(base, spec), f"{base.tolist()} vs {spec.tolist()}"
    runners = drivers[1].piecewise
    assert runners["mtp_trunk"].calls > 0
    assert runners["mtp_prefill"].calls == 1
    assert runners["mtp_draft_phase"].calls == runners["mtp_trunk"].calls
    assert runners["mtp_sync"].calls == 0
    if k >= 2:
        # the chain iterations after the prefill use mtp_draft
        assert runners["mtp_draft"].calls == k - 1


def test_mtp_three_graph_fallback_matches_baseline(monkeypatch):
    monkeypatch.setenv("MSTAR_GLM52_MTP_DRAFT_PHASE_GRAPH", "0")
    (base, spec), drivers = _run_pair(
        k=3, max_tokens=18, ignore_eos=True, mla_absorb=True, regions=True,
    )
    assert torch.equal(base, spec)
    runners = drivers[1].piecewise
    assert "mtp_draft_phase" not in runners
    assert runners["mtp_sync"].calls == runners["mtp_trunk"].calls
    assert runners["mtp_draft"].calls == 2 * (runners["mtp_trunk"].calls + 1)


def test_mtp_regions_batch_of_two_padded_to_four():
    """Two requests through regions captured at bs 4: two padding rows per
    replay (zero-length plan rows, sink-page scatter tails)."""
    cfg = _cfg(2, True)
    model = _model(cfg)
    prompts = {"a": torch.arange(5, dtype=torch.long) + 3, "b": torch.arange(3, dtype=torch.long) + 9}
    singles = {}
    for rid, prompt in prompts.items():
        sub = Glm52LLMSubmodule(model, cfg)
        singles[rid] = _drive(_Driver(sub, cfg, [rid]), prompt, _fwd_info(rid, 14, True))
    sub = Glm52LLMSubmodule(model, cfg)
    driver = _Driver(sub, cfg, list(prompts), regions=True, batch_sizes=(4,))
    infos = {rid: _fwd_info(rid, 14, True) for rid in prompts}
    emitted = {rid: [] for rid in prompts}
    texts = {}
    for rid, prompt in prompts.items():
        out = driver.step("prefill", {rid: (infos[rid], prompt)})[rid]
        emitted[rid].append(out["new_token"][0])
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


def test_last_rows_slices_padded_prefill_to_real_requests():
    # A captured prefill plan pads qo_indptr to the bucket and tail-fills it
    # with the last real offset. _last_rows must return one row per real
    # request, not one per bucket slot: a padded slot duplicates the final
    # request's row and hands the sampler more logit rows than it has
    # per-request params (an out-of-bounds read on the eager sampler).
    cfg = _cfg(0, mla_absorb=True)
    sub = Glm52LLMSubmodule(_model(cfg), cfg)
    # three real requests (lengths 5, 7, 3) captured in a bucket of four
    qo_indptr = torch.tensor([0, 5, 12, 15, 15])
    hidden = torch.randn(15, cfg.hidden_size)
    sub._attn = lambda _ei: SimpleNamespace(qo_indptr_buf=lambda _slot: qo_indptr)
    engine_inputs = SimpleNamespace(request_ids=["a", "b", "c"])
    rows = sub._last_rows(engine_inputs, hidden, {})
    assert rows.shape[0] == 3
    assert torch.equal(rows, hidden[[4, 11, 14]])


@pytest.mark.parametrize("k", [1, 2, 3])
def test_mtp_draft_phase_hoist_matches_baseline(monkeypatch, k):
    """MSTAR_GLM52_MTP_PHASE_PREPARE=1: sub-plan 0 and the sync inputs go in
    through runner.stage() before the verify readback, the chain sub-plans
    after; the stream is unchanged and every decode step stages once."""
    monkeypatch.setenv("MSTAR_GLM52_MTP_PHASE_PREPARE", "1")
    (base, spec), drivers = _run_pair(
        k=k, max_tokens=18, ignore_eos=True, mla_absorb=True, regions=True,
    )
    assert torch.equal(base, spec), f"{base.tolist()} vs {spec.tolist()}"
    runners = drivers[1].piecewise
    phase = runners["mtp_draft_phase"]
    # the first decode step (the emitted token alone, no bundle) is not k+1
    # rows and takes the un-hoisted path
    assert phase.calls - 1 <= phase.staged <= phase.calls and phase.staged > 0
