"""The MTP step's graph configs and host-side bookkeeping (CPU, reduced).

What test_glm52_engine_cycle.py does not already run: the piecewise config
inventory per k / flag (labels, PACKED buckets, ``declare_step`` callables,
static inputs, compile mode), the no-full-forward-graphs rule under MTP,
the region step declarations (trunk rows committed, draft phase's k
sub-plans uncommitted), preprocess's per-walk runner selection through
``can_run``, the acceptance log line, the load heartbeat's stop, and the
trunk-KV / plane bookkeeping between an MTP-off and an MTP-on run.
"""
from __future__ import annotations

import logging
import sys
import threading
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
    monkeypatch.setitem(sys.modules, "flashinfer", _cpu_flashinfer())


from mstar.engine.cuda_graph_config import PiecewiseConfigType  # noqa: E402
from mstar.engine.resources import MlaSubPlan, StepContext  # noqa: E402
from mstar.model.glm52._testing import build_cpu_resources  # noqa: E402
from mstar.model.glm52.components.causal_lm import Glm52ForCausalLM  # noqa: E402
from mstar.model.glm52.config import (  # noqa: E402
    ATTN_RESOURCE,
    KV_RESOURCE,
    SAMPLER_RESOURCE,
    Glm52ModelConfig,
)
from mstar.model.glm52.quantization import process_weights_after_loading  # noqa: E402
from mstar.model.glm52.submodules import (  # noqa: E402
    MTP_DRAFT_BUNDLE,
    MTP_DRAFT_LABEL,
    MTP_DRAFT_PHASE_LABEL,
    MTP_PREFILL_LABEL,
    MTP_SYNC_LABEL,
    MTP_TRUNK_LABEL,
    Glm52LLMSubmodule,
)
from mstar.model.submodule_base import ModelInputsFromEngine  # noqa: E402

CPU = torch.device("cpu")


def _mtp_cfg(k: int) -> Glm52ModelConfig:
    cfg = Glm52ModelConfig.reduced()
    cfg.num_hidden_layers = 4  # the MTP position lands FULL (4 = offset-1 + freq)
    cfg.mtp_num_draft_tokens = k
    cfg.mla_absorb = True  # the draft-phase graph needs the MLA resource's sub-plans
    return cfg


def _model(cfg: Glm52ModelConfig, seed: int = 0) -> Glm52ForCausalLM:
    """Every parameter randomized: the MoE expert containers are raw
    ``torch.empty`` at construction (the loader fills them), and garbage
    there NaNs the logits — a NaN model emits an all-zero argmax stream on
    every path, which makes any stream comparison vacuous."""
    torch.manual_seed(seed)
    model = Glm52ForCausalLM(cfg)
    for name, p in model.named_parameters():
        if "norm" in name:
            p.data.normal_(1.0, 0.02)
        elif name.endswith("gate.weight") or "e_score_correction_bias" in name:
            p.data.normal_(0, 1.0)
        else:
            p.data.normal_(0, 0.05)
    process_weights_after_loading(model, CPU)
    return model.eval()


def _fwd_info(rid: str, max_tokens: int, ignore_eos: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=rid,
        max_tokens=max_tokens,
        resource_configs={SAMPLER_RESOURCE: SimpleNamespace(
            ignore_eos=ignore_eos, temperature=0.0, repetition_penalty=1,
        )},
        dynamic_loop_iter_counts={},
    )


def _piecewise(sub: Glm52LLMSubmodule):
    return sub.get_piecewise_cuda_graph_configs(CPU, torch.bfloat16, tp_world_size=1)


# ── config inventory ──


def test_mtp_disables_full_forward_capture_configs():
    """With MTP on, no full-forward CUDA-graph configs may register: their
    warmup captures crash (host-side verify/rewind; packed prefill never
    runs preprocess) and the failure mode is a silent 13x eager fallback.
    Flag off keeps the decode + prefill capture pair."""
    cfg = _mtp_cfg(2)
    sub = Glm52LLMSubmodule(Glm52ForCausalLM(cfg), cfg)
    assert sub.get_cuda_graph_configs(CPU) == []
    cfg.mtp_num_draft_tokens = 0
    assert len(sub.get_cuda_graph_configs(CPU)) == 2
    assert _piecewise(sub) == {}


def test_mtp_trunk_piecewise_config_shapes():
    """The trunk graph registers exactly one (bs, [k+1]*bs) PACKED bucket
    per capture batch size — never the bs x token-bucket cross product —
    and k >= 2 adds the 1-row-per-request draft-chain graph. Every region
    declares its own resource step (``declare_step``), and the compile mode
    is the config's ``"default"`` (the cuBLAS one: 90.03 -> 96.97 tok/s TP8)."""
    cfg = _mtp_cfg(2)  # rows per request = 3
    sub = Glm52LLMSubmodule(Glm52ForCausalLM(cfg), cfg)
    # Sync capture is env-default-ON as of 2026-08-11 (measured clean, 3264
    # bit-exact, arm C). Pin it OFF here to check the trunk+draft-only set,
    # then ON below for the sync-included set.
    sub._mtp_capture_sync = False
    sub._mtp_capture_prefill = False
    sub._mtp_draft_phase_graph = False
    assert set(_piecewise(sub)) == {MTP_TRUNK_LABEL, MTP_DRAFT_LABEL}
    sub._mtp_capture_sync = True
    configs = _piecewise(sub)
    assert set(configs) == {MTP_TRUNK_LABEL, MTP_DRAFT_LABEL, MTP_SYNC_LABEL}
    # The one-graph draft phase (2026-08-19): sync capture on + flag on adds
    # a (bs, k+1) PACKED bucket per batch size whose step declares the k
    # attention sub-plans (resource-owned; its e_list arrives per replay
    # through run(step_kwargs=)), and the caller inputs.
    sub._mtp_draft_phase_graph = True
    configs = _piecewise(sub)
    assert set(configs) == {
        MTP_TRUNK_LABEL, MTP_DRAFT_LABEL, MTP_SYNC_LABEL, MTP_DRAFT_PHASE_LABEL}
    ph = configs[MTP_DRAFT_PHASE_LABEL]
    assert ph.get_config_type() == PiecewiseConfigType.PACKED
    assert ph.declare_step == sub._draft_phase_region_step
    phshapes = ph.get_capture_shapes(ph.capture_batch_sizes)
    assert [(s.bs, s.total_tokens) for s in phshapes] == [
        (1, 3), (2, 6), (4, 12), (8, 24), (16, 48)]
    phstatic = ph.make_static_inputs(phshapes[1])
    assert set(phstatic) == {"sync_ids", "pair_hidden", "sync_position_ids",
                             "last_rows", "chain_pos_1"}   # k=2: one chain iteration
    assert phstatic["last_rows"].shape == (2,)
    assert phstatic["chain_pos_1"].shape == (2,)  # per-request, per iteration
    assert phstatic["pair_hidden"].dtype == torch.bfloat16
    # the naive backend has no sub-plans: no draft-phase graph
    cfg.mla_absorb = False
    assert MTP_DRAFT_PHASE_LABEL not in _piecewise(sub)
    cfg.mla_absorb = True
    sub._mtp_draft_phase_graph = False
    # The captured MTP prefill (2026-08-19) is env-default-ON and adds the
    # k=0 config's packed prefill buckets — bs x token-bucket, PACKED, the
    # full-row hidden/prenorm outputs — never lm_head.
    sub._mtp_capture_prefill = True
    configs = _piecewise(sub)
    assert set(configs) == {
        MTP_TRUNK_LABEL, MTP_DRAFT_LABEL, MTP_SYNC_LABEL, MTP_PREFILL_LABEL}
    pf = configs[MTP_PREFILL_LABEL]
    assert pf.get_config_type() == PiecewiseConfigType.PACKED
    assert pf.declare_step == sub._trunk_region_step
    pshapes = pf.get_capture_shapes(pf.capture_batch_sizes)
    buckets = cfg.prefill_token_buckets or sub.PREFILL_TOKEN_BUCKETS
    bss = cfg.prefill_capture_batch_sizes or sub.PREFILL_CAPTURE_BATCH_SIZES
    assert sorted((s.bs, s.total_tokens) for s in pshapes) == sorted(
        (b, t) for b in bss for t in buckets)
    assert all(sum(s.seq_lens) == s.total_tokens for s in pshapes)
    pstatic = pf.make_static_inputs(pshapes[0])
    assert set(pstatic) == {"input_ids", "position_ids"}
    assert pstatic["input_ids"].shape == (pshapes[0].total_tokens,)
    # replay pads absent requests with zero-length plan rows
    assert pf.replay_seq_lens(pshapes[0], [5], 1) == [5]

    pc = configs[MTP_TRUNK_LABEL]
    assert pc.get_config_type() == PiecewiseConfigType.PACKED
    assert pc.declare_step == sub._trunk_region_step
    assert pc.rows_per_request == 3
    shapes = pc.get_capture_shapes(pc.capture_batch_sizes)
    assert [(s.bs, s.total_tokens) for s in shapes] == [
        (1, 3), (2, 6), (4, 12), (8, 24), (16, 48)]
    assert all(s.seq_lens == [3] * s.bs for s in shapes)
    assert pc.replay_seq_lens(shapes[2], [3, 3], 2) == [3, 3, 0, 0]
    with pytest.raises(ValueError, match="seq_lens"):
        pc.replay_seq_lens(shapes[0], None, 1)
    static = pc.make_static_inputs(shapes[0])
    assert set(static) == {"input_ids", "position_ids"}
    assert static["input_ids"].shape == (3,)
    assert static["input_ids"].dtype == torch.long

    dc = configs[MTP_DRAFT_LABEL]
    assert dc.get_config_type() == PiecewiseConfigType.PACKED
    assert dc.declare_step == sub._chain_region_step
    dshapes = dc.get_capture_shapes(dc.capture_batch_sizes)
    assert [(s.bs, s.total_tokens) for s in dshapes] == [
        (1, 1), (2, 2), (4, 4), (8, 8), (16, 16)]
    dstatic = dc.make_static_inputs(dshapes[1])
    assert set(dstatic) == {"draft_ids", "prev_hidden", "position_ids"}
    assert dstatic["draft_ids"].shape == (2,)
    assert dstatic["prev_hidden"].shape == (2, cfg.hidden_size)
    assert dstatic["prev_hidden"].dtype == torch.bfloat16

    # The padded sync graph shares the trunk's capture shape exactly.
    sc = configs[MTP_SYNC_LABEL]
    assert sc.get_config_type() == PiecewiseConfigType.PACKED
    assert sc.declare_step == sub._trunk_region_step
    sshapes = sc.get_capture_shapes(sc.capture_batch_sizes)
    assert [(s.bs, s.total_tokens) for s in sshapes] == [
        (1, 3), (2, 6), (4, 12), (8, 24), (16, 48)]
    sstatic = sc.make_static_inputs(sshapes[0])
    assert set(sstatic) == {"sync_ids", "pair_hidden", "position_ids"}
    assert sstatic["sync_ids"].shape == (3,)
    assert sstatic["pair_hidden"].shape == (3, cfg.hidden_size)
    assert sstatic["pair_hidden"].dtype == torch.bfloat16

    # every region compiles in the cuBLAS mode; the capture_fn is the
    # region's own method
    for label, c in configs.items():
        assert c.compile is True and c.compile_mode == "default", label
        assert callable(c.capture_fn), label

    # k=1 has no chain iterations — no draft graph to pay capture for —
    # but the sync pass (rows=2) still registers when opted in.
    cfg.mtp_num_draft_tokens = 1
    assert set(_piecewise(sub)) == {MTP_TRUNK_LABEL, MTP_SYNC_LABEL, MTP_PREFILL_LABEL}
    sub._mtp_draft_phase_graph = True  # k=1: sync + draft-1 head, no chain
    k1 = _piecewise(sub)
    assert MTP_DRAFT_PHASE_LABEL in k1
    assert set(k1[MTP_DRAFT_PHASE_LABEL].make_static_inputs(
        k1[MTP_DRAFT_PHASE_LABEL].get_capture_shapes([1])[0])) == {
            "sync_ids", "pair_hidden", "sync_position_ids", "last_rows"}
    sub._mtp_draft_phase_graph = False
    sub._mtp_capture_sync = False

    cfg.mtp_num_draft_tokens = 0
    assert _piecewise(sub) == {}
    cfg.mtp_num_draft_tokens = 2


def test_graph_compile_escape_hatch_reaches_the_regions(monkeypatch):
    cfg = _mtp_cfg(2)
    sub = Glm52LLMSubmodule(Glm52ForCausalLM(cfg), cfg)
    monkeypatch.setenv("MSTAR_GLM52_GRAPH_COMPILE", "0")
    assert not any(c.compile for c in _piecewise(sub).values())


# ── region step declarations ──


def _bound(cfg: Glm52ModelConfig, rids=("r0",)):
    sub = Glm52LLMSubmodule(_model(cfg), cfg)
    resources, runner = build_cpu_resources(cfg, list(rids))
    sub.bind_node_resources(resources)
    return sub, resources, runner


def _run_step(runner, step, rids):
    step.set_ctx(StepContext(request_ids=tuple(rids), graph_walk="decode", slot=0, capture=False))
    assert runner.admit(step).ok
    runner.plan(step)
    runner.commit(step)


def test_region_steps_trunk_commits_draft_phase_declares_sub_plans():
    """The trunk / sync / prefill regions append and commit their rows; the
    draft phase declares k attention sub-plans over the same stream from
    ``e_list`` — the padded sync pass at P0 = stored - e over k+1 rows,
    then one chain row per iteration — and commits nothing; capture (no
    e_list) plans e=0 at the stored length."""
    k = 2
    cfg = _mtp_cfg(k)
    sub, resources, runner = _bound(cfg)
    kv = resources[KV_RESOURCE]

    trunk = sub._trunk_region_step(["r0"], [5])
    assert set(trunk.keys()) == {KV_RESOURCE, ATTN_RESOURCE}
    assert trunk.get(KV_RESOURCE).commit is True
    assert trunk.get(ATTN_RESOURCE).sub_plans is None
    assert trunk.segments[0].span == 5
    _run_step(runner, trunk, ["r0"])
    assert kv.stored_len("r0") == 5

    chain = sub._chain_region_step(["r0"], [1])
    assert chain.get(KV_RESOURCE).commit is True and chain.segments[0].span == 1

    # replay: the trunk committed k+1 rows and the verify rewound k+1-e
    phase = sub._draft_phase_region_step(["r0"], [k + 1], e_list=[2])
    assert phase.get(KV_RESOURCE).commit is False
    assert phase.segments[0].span == max(k + 1 - 2, k - 1) == 1
    assert phase.get(ATTN_RESOURCE).sub_plans == (
        MlaSubPlan(q_lens=(k + 1,), kv_lens=(3 + k + 1,)),  # P0 = 5 - 2
        MlaSubPlan(q_lens=(1,), kv_lens=(5 + 1,)),
    )
    _run_step(runner, phase, ["r0"])
    assert kv.stored_len("r0") == 5  # nothing committed

    # capture / warmup: no e_list, fresh-stream semantics at the length
    capture = sub._draft_phase_region_step(["r0"], [k + 1])
    assert capture.segments[0].span == k + 1
    assert capture.get(ATTN_RESOURCE).sub_plans == (
        MlaSubPlan(q_lens=(k + 1,), kv_lens=(5 + k + 1,)),
        MlaSubPlan(q_lens=(1,), kv_lens=(5 + 1,)),
    )

    # a padding row (seq_len 0) is absent from every sub-plan and gets no e
    padded = sub._draft_phase_region_step(["r0", "pad"], [k + 1, 0], e_list=[1])
    assert padded.segments[1].span == 0
    assert padded.get(ATTN_RESOURCE).sub_plans == (
        MlaSubPlan(q_lens=(k + 1, 0), kv_lens=(4 + k + 1, 0)),
        MlaSubPlan(q_lens=(1, 0), kv_lens=(6, 0)),
    )

    # the naive backend cannot host sub-plans
    cfg.mla_absorb = False
    with pytest.raises(RuntimeError, match="sub-plans"):
        sub._draft_phase_region_step(["r0"], [k + 1], e_list=[1])


# ── preprocess: runner selection ──


class _StubRunner:
    """A piecewise runner as preprocess sees it: ``can_run`` for a shape."""

    def __init__(self, ok: bool = True):
        self.ok = ok
        self.asked: list[tuple[int, int]] = []

    def can_run(self, batch_size, total_tokens=None):
        self.asked.append((batch_size, total_tokens))
        return self.ok


def _preprocess(sub, resources, walk, seq_lens, runners):
    from mstar.model.submodule_base import ARNodeInputs

    rids = [f"r{i}" for i in range(len(seq_lens))]
    inputs = [
        ARNodeInputs(input_ids=torch.zeros(n, dtype=torch.long), input_seq_len=n)
        for n in seq_lens
    ]
    engine_inputs = ModelInputsFromEngine(
        request_ids=rids, per_request_info={}, resources=resources,
        piecewise_runners=runners,
    )
    return sub.preprocess(walk, engine_inputs, inputs)


def test_preprocess_selects_runners_per_walk_through_can_run():
    """The replay decision is made once, in preprocess, per region: a
    runner rides ``mtp_runners`` only if it has a bucket for THIS shape —
    trunk / prefill asked for the real token count, sync and draft phase
    for the padded (k+1) x bs rows, the chain for one row per request.
    Prefill takes the prefill trunk and the chain; decode the trunk, sync,
    phase and chain."""
    k = 2
    cfg = _mtp_cfg(k)
    sub, resources, _ = _bound(cfg, ["r0", "r1"])
    runners = {label: _StubRunner() for label in (
        MTP_TRUNK_LABEL, MTP_DRAFT_LABEL, MTP_SYNC_LABEL, MTP_DRAFT_PHASE_LABEL,
        MTP_PREFILL_LABEL)}

    kw = _preprocess(sub, resources, "decode", [k + 1, k + 1], runners)
    got = kw["mtp_runners"]
    assert set(got) == {"trunk", "sync", "phase", "draft"}
    assert got["trunk"] is runners[MTP_TRUNK_LABEL]
    assert got["sync"] is runners[MTP_SYNC_LABEL]
    assert got["phase"] is runners[MTP_DRAFT_PHASE_LABEL]
    assert got["draft"] is runners[MTP_DRAFT_LABEL]
    assert runners[MTP_TRUNK_LABEL].asked == [(2, 6)]
    assert runners[MTP_SYNC_LABEL].asked == [(2, 6)]
    assert runners[MTP_DRAFT_PHASE_LABEL].asked == [(2, 6)]
    assert runners[MTP_DRAFT_LABEL].asked == [(2, 2)]
    assert runners[MTP_PREFILL_LABEL].asked == []
    assert kw["seq_lens"] == [3, 3]
    assert kw["position_ids"].tolist() == [0, 1, 2, 0, 1, 2]

    for r in runners.values():
        r.asked.clear()
    kw = _preprocess(sub, resources, "prefill", [5, 3], runners)
    got = kw["mtp_runners"]
    assert set(got) == {"prefill", "draft"}
    assert got["prefill"] is runners[MTP_PREFILL_LABEL]
    assert got["draft"] is runners[MTP_DRAFT_LABEL]
    assert runners[MTP_PREFILL_LABEL].asked == [(2, 8)]
    assert runners[MTP_DRAFT_LABEL].asked == [(2, 2)]
    assert runners[MTP_TRUNK_LABEL].asked == []
    assert torch.equal(kw["last_token_indices"], torch.tensor([4, 7]))

    # a region without a bucket for the shape runs eager: None in the dict
    refusing = {MTP_TRUNK_LABEL: _StubRunner(ok=False), MTP_DRAFT_LABEL: _StubRunner()}
    got = _preprocess(sub, resources, "decode", [k + 1], refusing)["mtp_runners"]
    assert got["trunk"] is None and got["sync"] is None and got["phase"] is None
    assert got["draft"] is refusing[MTP_DRAFT_LABEL]
    # no runners at all (eager serving): every region None
    got = _preprocess(sub, resources, "decode", [k + 1], {})["mtp_runners"]
    assert got == {"trunk": None, "sync": None, "phase": None, "draft": None}
    got = _preprocess(sub, resources, "decode", [k + 1], None)["mtp_runners"]
    assert set(got) == {"trunk", "sync", "phase", "draft"}
    # k=0: no regions, nothing asked
    cfg.mtp_num_draft_tokens = 0
    assert _preprocess(sub, resources, "decode", [1], runners)["mtp_runners"] == {}


# ── acceptance log ──


def test_mtp_acceptance_log_per_position(caplog):
    """The 512-step acceptance line must carry the conditional per-position
    profile (the datum that separates "first draft mediocre" from "chained
    drafts collapse"). Short tests never cross the threshold, so drive the
    method directly with a synthetic histogram."""
    k = 3

    def _ns(pair_postnorm: bool) -> SimpleNamespace:
        return SimpleNamespace(
            config=SimpleNamespace(mtp_num_draft_tokens=k),
            _MTP_STAT_LOG_EVERY=Glm52LLMSubmodule._MTP_STAT_LOG_EVERY,
            _mtp_stat_steps=512,
            _mtp_stat_logged=0,
            # 512 steps halving at each position: reached = [512, 256, 128, 64].
            _mtp_stat_acc_hist=[256, 128, 64, 64],
            _mtp_stat_emitted=256 * 1 + 128 * 2 + 64 * 3 + 64 * 4,
            _mtp_pair_postnorm=pair_postnorm,
        )

    ns = _ns(False)
    with caplog.at_level(logging.INFO, logger="mstar.model.glm52.submodules"):
        Glm52LLMSubmodule._maybe_log_mtp_acceptance(ns)
    msgs = [r.getMessage() for r in caplog.records]
    assert any("emitted/step" in m for m in msgs)
    (pos_line,) = [m for m in msgs if "by position" in m]
    assert "[256, 128, 64, 64]" in pos_line
    assert "0.50 0.50 0.50" in pos_line
    assert ns._mtp_stat_logged == 512
    # The line must name which trunk-pairing arm produced it: an acceptance
    # profile whose arm you infer from launch env is one you cannot trust
    # after the fact, and mislabelling an arm silently inverts the A/B.
    assert "pre-final-norm" in pos_line and "POST" not in pos_line

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="mstar.model.glm52.submodules"):
        Glm52LLMSubmodule._maybe_log_mtp_acceptance(_ns(True))
    (post_line,) = [
        r.getMessage() for r in caplog.records if "by position" in r.getMessage()
    ]
    assert "POST-final-norm" in post_line
    # below the threshold nothing is logged
    caplog.clear()
    quiet = _ns(True)
    quiet._mtp_stat_steps = 100
    with caplog.at_level(logging.INFO, logger="mstar.model.glm52.submodules"):
        Glm52LLMSubmodule._maybe_log_mtp_acceptance(quiet)
    assert not caplog.records


# ── load heartbeat ──


class _Driver:
    """The engine's per-step cycle for one node over the real resources
    (eager regions: every MTP phase runs through ``_eager_step``)."""

    def __init__(self, sub: Glm52LLMSubmodule, cfg: Glm52ModelConfig, rids: list[str]):
        self.sub = sub
        self.resources, self.runner = build_cpu_resources(cfg, rids)
        sub.bind_node_resources(self.resources)

    def step(self, walk: str, batch: dict[str, tuple[SimpleNamespace, torch.Tensor]]):
        rids = list(batch)
        inputs = [
            self.sub.prepare_inputs(walk, info, {"text_inputs": [text]})
            for info, text in batch.values()
        ]
        step = self.sub.declare_step(walk, rids, inputs)
        if step is not None:
            step.set_ctx(StepContext(request_ids=tuple(rids), graph_walk=walk, slot=0, capture=False))
            assert self.runner.admit(step).ok
            self.runner.plan(step)
        engine_inputs = ModelInputsFromEngine(
            request_ids=rids,
            per_request_info={rid: info for rid, (info, _) in batch.items()},
            resources=self.resources, step=step,
        )
        kw = self.sub.preprocess(walk, engine_inputs, inputs)
        outs = self.sub.forward_batched(walk, engine_inputs, **kw)
        if step is not None:
            self.runner.commit(step)
        for rid, (info, _) in batch.items():
            self.sub.postprocess(rid, info, outs[rid])
        return outs


def _drive(driver: _Driver, prompt: torch.Tensor, info, max_steps=64) -> torch.Tensor:
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
        walk, text = "decode", out["text_inputs"][0]
    return torch.cat(emitted)


def test_graph_config_getters_stop_the_load_heartbeat():
    """Both capture-config getters stop the load heartbeat FIRST: the tick
    is a foreign-thread CUDA kernel, illegal during a (global-mode) graph
    capture, and capture starts right after the configs are read. The
    keeper process covers the box reaper across capture (it used to be the
    tick + thread_local capture mode). A forward stops it too, for the
    eager-only paths that never capture."""
    cfg = _mtp_cfg(2)
    model = Glm52ForCausalLM(cfg)

    sub = Glm52LLMSubmodule(model, cfg)
    stop = threading.Event()
    sub.set_load_heartbeat_stop(stop)
    sub.get_piecewise_cuda_graph_configs(CPU, torch.bfloat16)
    assert stop.is_set(), "piecewise config getter must stop the heartbeat"
    assert sub._load_heartbeat_stop is None

    sub2 = Glm52LLMSubmodule(model, cfg)
    stop2 = threading.Event()
    sub2.set_load_heartbeat_stop(stop2)
    sub2.get_cuda_graph_configs(CPU)
    assert stop2.is_set(), "full-graph config getter must stop the heartbeat"
    assert sub2._load_heartbeat_stop is None

    # eager-only serving never reads a config: the first forward stops it
    cfg.mtp_num_draft_tokens = 0
    sub3 = Glm52LLMSubmodule(_model(cfg), cfg)
    stop3 = threading.Event()
    sub3.set_load_heartbeat_stop(stop3)
    driver = _Driver(sub3, cfg, ["r0"])
    driver.step("prefill", {"r0": (_fwd_info("r0", 4), torch.tensor([1, 2, 3]))})
    assert stop3.is_set()
    # idempotent once stopped (and on a submodule that never had one)
    sub3._stop_load_heartbeat()
    Glm52LLMSubmodule._stop_load_heartbeat(object.__new__(Glm52LLMSubmodule))


# ── KV bookkeeping between the MTP-off and MTP-on runs ──


def _latent_rows(kv, rid: str, layer: int, n: int) -> torch.Tensor:
    pages = kv._streams[rid]["main"].page_indices
    return kv.layer_view(layer)[pages].reshape(-1, kv.config.head_dim)[:n]


def test_mtp_trunk_kv_and_plane_bookkeeping():
    """Same weights, MTP off then on: the streams match, and the trunk's
    latent cache for the verified stream matches row for row — accepted
    rows are recomputed for the same tokens at the same positions,
    rejected tails were rewound and sit above the length. The prompt rows
    (one prefill of the same shape in both runs) are bitwise equal; decode
    rows come from 1-row steps vs (k+1)-row verify batches, whose fp32
    op-order residual (~1e-7) can flip the bf16 rounding inside the
    engine's ``flashinfer_rmsnorm`` op by one ulp (~8e-3 at these
    magnitudes), so they are pinned to that bound. The MTP plane exists
    only in the flag-on run (one extra KV layer) and holds an entry for
    every stored position."""
    cfg = _mtp_cfg(2)
    model = _model(cfg)
    prompt = torch.arange(5, dtype=torch.long) + 3
    streams, kvs = [], []
    for mode_k in (0, 2):
        cfg.mtp_num_draft_tokens = mode_k
        sub = Glm52LLMSubmodule(model, cfg)
        driver = _Driver(sub, cfg, ["r0"])
        streams.append(_drive(driver, prompt, _fwd_info("r0", 16)))
        kvs.append(driver.resources[KV_RESOURCE])
    cfg.mtp_num_draft_tokens = 2
    base, spec = streams
    assert torch.equal(base, spec) and base.numel() == 16
    assert len(set(base.tolist())) > 1  # a real (non-degenerate) stream

    off, on = kvs
    assert off.config.num_layers == cfg.num_hidden_layers
    assert on.config.num_layers == cfg.num_hidden_layers + 1
    # both caches hold the prompt plus every emitted token but the last
    n = prompt.numel() + 16 - 1
    assert off.stored_len("r0") == on.stored_len("r0") == n
    for layer in range(cfg.num_hidden_layers):
        rows_on = _latent_rows(on, "r0", layer, n)
        rows_off = _latent_rows(off, "r0", layer, n)
        assert torch.isfinite(rows_on).all() and rows_on.abs().sum(-1).gt(0).all()
        assert torch.equal(rows_on[:prompt.numel()], rows_off[:prompt.numel()]), layer
        torch.testing.assert_close(
            rows_on, rows_off, atol=3e-2, rtol=1e-2, msg=f"layer {layer} KV diverged",
        )
    plane = _latent_rows(on, "r0", cfg.num_hidden_layers, n)
    assert torch.isfinite(plane).all() and plane.abs().sum(-1).gt(0).all()


@pytest.mark.parametrize("accepted", [2, 1, 0], ids=["all", "first", "none"])
def test_mtp_verify_and_rewind_under_forced_acceptance(accepted):
    """Random weights draft ~nothing, so the stream tests never see an
    accepted draft. Replace the draft phase with an oracle that reads the
    baseline stream and proposes a fixed number of correct drafts per step:
    every step then keeps e = accepted + 1 rows of its k+1 committed
    (``KVManager.rewind`` by the rest), the emission is e tokens, the
    acceptance histogram lands entirely in that bin, and the stream is
    still the baseline's, bit for bit."""
    k = 2
    cfg = _mtp_cfg(k)
    model = _model(cfg, seed=1)
    prompt = torch.arange(5, dtype=torch.long) + 3
    max_tokens = 19

    cfg.mtp_num_draft_tokens = 0
    base = _drive(_Driver(Glm52LLMSubmodule(model, cfg), cfg, ["r0"]), prompt, _fwd_info("r0", max_tokens))
    assert base.numel() == max_tokens

    cfg.mtp_num_draft_tokens = k
    sub = Glm52LLMSubmodule(model, cfg)
    driver = _Driver(sub, cfg, ["r0"])
    kv = driver.resources[KV_RESOURCE]
    wrong = torch.tensor(cfg.vocab_size - 1)

    def oracle(engine_inputs, sync_tokens, pair_hiddens, **kwargs):
        # the tokens emitted so far index the baseline: the next ``accepted``
        # of them are right, the rest deliberately wrong
        m = sub._mtp_emitted["r0"]
        drafts = [
            base[m + j] if j < accepted and m + j < base.numel() else wrong
            for j in range(k)
        ]
        return [torch.stack(drafts)]

    sub._mtp_sync_and_draft = oracle

    rid, info = "r0", _fwd_info("r0", max_tokens)
    emitted = []
    out = driver.step("prefill", {rid: (info, prompt)})[rid]
    emitted.append(out["new_token"][0])
    # carry the prefill's [emitted, k drafts] bundle (the default-on edge),
    # so the first decode step is a (k+1)-row step like every other
    text = out[MTP_DRAFT_BUNDLE][0]
    assert torch.equal(text[:1], out["new_token"][0])
    total = prompt.numel()
    e_seen = []
    for _ in range(64):
        assert text.numel() == k + 1
        out = driver.step("decode", {rid: (info, text)})[rid]
        e = out["new_token"][0].numel()
        e_seen.append(e)
        emitted.append(out["new_token"][0])
        total += e
        # k+1 rows committed, k+1-e rewound: the stream holds the emission
        assert kv.stored_len(rid) == total
        if sub.check_stop(rid, info, out):
            break
        text = out["text_inputs"][0]
    stream = torch.cat(emitted)
    assert torch.equal(stream, base), f"{stream.tolist()} vs {base.tolist()}"
    # every non-final step emitted accepted+1; the last may be truncated
    assert all(e == accepted + 1 for e in e_seen[:-1]) and e_seen[-1] <= accepted + 1
    hist = sub._mtp_stat_acc_hist
    assert hist[accepted] == sub._mtp_stat_steps == len(e_seen)
    assert sum(hist) == hist[accepted]
