"""M3 draft-loop integration tests (CPU, reduced config).

THE property: at temp 0, MTP-on emits the bit-identical token stream to
MTP-off — greedy verify guarantees it by construction, and these tests
execute the guarantee through the real submodule protocol
(prepare_inputs → preprocess → forward_batched → postprocess → check_stop)
with ``ReferenceCacheHandle`` standing in for the engine. Draft quality is
irrelevant to the property (random weights draft ~nothing); acceptance
length is a box measurement, not a CPU test.
"""

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


# Import-time stub so this module imports on machines without CUDA wheels.
if "flashinfer" not in sys.modules:
    sys.modules["flashinfer"] = _cpu_flashinfer()


@pytest.fixture(autouse=True)
def _force_cpu_flashinfer(monkeypatch):
    """Force the CPU stub per test — see test_glm52_mtp.py for the why.

    THE property this file exists to prove (MTP-on emits the bit-identical
    stream to MTP-off) was silently not being proven in full-suite runs on
    the box: real flashinfer is installed there, an earlier-imported module
    put it in sys.modules, the import-time guard then skipped the stub, and
    these tests ran GPU kernels on CPU tensors. Eight tests failed in the
    suite while every one passed in isolation. A bit-identity suite that
    quietly stops checking bit-identity is this lane's whole failure class,
    so the stub is now forced rather than deferred-to."""
    monkeypatch.setitem(sys.modules, "flashinfer", _cpu_flashinfer())

from mstar.model.glm52._testing import ReferenceCacheHandle  # noqa: E402
from mstar.model.glm52.components.causal_lm import Glm52ForCausalLM  # noqa: E402
from mstar.model.glm52.config import Glm52ModelConfig  # noqa: E402
from mstar.model.glm52.submodules import Glm52LLMSubmodule  # noqa: E402


class _ArgmaxSampler:
    """Greedy stand-in for the engine sampler (temp 0, no penalties)."""

    def sample(self, request_ids, logits, apply_penalty=True):
        return logits.argmax(dim=-1)


class _EngineInputs:
    def __init__(self, cache_manager, sampler, piecewise_runners=None):
        self.cache_manager = cache_manager
        self.sampler = sampler
        self.piecewise_runners = piecewise_runners or {}


class _StubTrunkRunner:
    """Contract double for ``PiecewiseCudaGraphRunner.run`` on the MTP
    trunk: plans on the (real) handle, runs the trunk eagerly, returns
    cloned row-sliced outputs — everything the submodule may rely on from
    a replay, with none of the CUDA. The submodule must produce a
    bit-identical stream through this seam vs the fully eager path."""

    def __init__(self, sub, handle):
        self.sub = sub
        self.handle = handle
        self.calls = 0

    def can_run(self, batch_size, total_tokens=None):
        return True

    def run(self, static_inputs, request_ids=None, seq_lens=None, real_bs=None):
        self.calls += 1
        # preprocess skipped its plan (replay decision) — the runner owns it.
        self.handle.plan_attention(seq_lens=seq_lens, is_causal=True, label="main")
        hidden, prenorm = self.sub._hidden(
            static_inputs["input_ids"], static_inputs["position_ids"],
            self.handle, with_prenorm=True)
        return {
            "hidden": hidden.clone(),
            "prenorm": prenorm.clone(),
            "logits": self.sub.lm_head(hidden).clone(),
        }


class _StubDraftRunner:
    """Contract double for the ``mtp_draft`` piecewise graph: one chain
    iteration — plan on the plane, rope at counter+1 derived from the
    aliased states (asserted equal to the caller's positions: the property
    the real plan_fn depends on), fused MTP forward, head argmax, advance
    — with cloned outputs, mimicking PiecewiseOutput's clone-on-index."""

    def __init__(self, sub, handle):
        self.sub = sub
        self.handle = handle
        self.calls = 0

    def can_run(self, batch_size, total_tokens=None):
        return True

    def run(self, static_inputs, request_ids=None, seq_lens=None, real_bs=None):
        self.calls += 1
        sub, handle = self.sub, self.handle
        handle.set_layer_idx(sub.config.num_hidden_layers)
        handle.plan_attention(seq_lens=seq_lens, is_causal=True, label="main")
        pos = torch.tensor(
            [handle._get_state(rid, "main").position_id_start + 1
             for rid in request_ids], dtype=torch.long)
        assert torch.equal(pos, static_inputs["position_ids"].cpu()), (
            "replay plan positions (counter+1) diverge from the loop's "
            f"st+it: {pos.tolist()} vs "
            f"{static_inputs['position_ids'].tolist()}")
        handle.plan_rope(seq_lens=seq_lens, pos_ids=pos, label="main")
        mtp = sub.language_model.mtp
        embed = sub.language_model.model.embed_tokens
        h_head, h_raw = mtp(
            embed(static_inputs["draft_ids"]), static_inputs["prev_hidden"],
            handle, pos)
        handle.advance_seq_lens()
        return {
            "draft_ids": sub.lm_head(h_head).argmax(dim=-1).clone(),
            "prev_hidden": h_raw.clone(),
        }


class _StubSyncRunner:
    """Contract double for the ``mtp_sync`` piecewise graph: the PADDED
    sync pass — k+1 rows per request, real rows first. Plans on the plane
    with rope at counter+1+row derived from the aliased states (asserted
    equal to the caller's padded positions: the contiguity property the
    real plan_fn depends on — it must need no knowledge of e), runs the
    fused MTP forward over ALL rows including pads (so pad entries land in
    the plane store exactly as a replay would write them), advances the
    full k+1, and returns cloned outputs. The caller owns the rows-e
    rewind correction and the last-real-row gather."""

    def __init__(self, sub, handle):
        self.sub = sub
        self.handle = handle
        self.calls = 0

    def can_run(self, batch_size, total_tokens=None):
        return True

    def run(self, static_inputs, request_ids=None, seq_lens=None, real_bs=None):
        self.calls += 1
        sub, handle = self.sub, self.handle
        handle.set_layer_idx(sub.config.num_hidden_layers)
        handle.plan_attention(seq_lens=seq_lens, is_causal=True, label="main")
        pos_l = []
        for rid, sl in zip(request_ids, seq_lens, strict=True):
            start = handle._get_state(rid, "main").position_id_start
            pos_l.extend(range(start + 1, start + 1 + sl))
        pos = torch.tensor(pos_l, dtype=torch.long)
        assert torch.equal(pos, static_inputs["position_ids"].cpu()), (
            "padded-sync plan positions (counter+1+row) diverge from the "
            f"layout's: {pos.tolist()} vs "
            f"{static_inputs['position_ids'].tolist()}")
        handle.plan_rope(seq_lens=seq_lens, pos_ids=pos, label="main")
        mtp = sub.language_model.mtp
        embed = sub.language_model.model.embed_tokens
        h_head, h_raw = mtp(
            embed(static_inputs["sync_ids"]), static_inputs["pair_hidden"],
            handle, pos)
        handle.advance_seq_lens()
        return {"h_head": h_head.clone(), "h_raw": h_raw.clone()}


def _mtp_cfg(k: int) -> Glm52ModelConfig:
    cfg = Glm52ModelConfig.reduced()
    cfg.num_hidden_layers = 4  # MTP position lands FULL (4 = offset-1 + freq)
    cfg.mtp_num_draft_tokens = k
    return cfg


def _fwd_info(max_tokens: int, ignore_eos: bool) -> SimpleNamespace:
    # Models the full SamplingConfig contract: the MTP prepare_inputs
    # guard reads temperature/repetition_penalty (greedy-only refusal).
    return SimpleNamespace(
        request_id="r0",
        max_tokens=max_tokens,
        sampling_config={"LLM": SimpleNamespace(
            ignore_eos=ignore_eos, temperature=0.0, repetition_penalty=1,
        )},
        dynamic_loop_iter_counts={},
    )


def _drive(sub, handle, prompt, fwd_info, max_steps=64, runners=None):
    """Run prefill + decode through the submodule protocol until stop."""
    ei = _EngineInputs(handle, _ArgmaxSampler(), piecewise_runners=runners)
    emitted = []
    walk, text_inputs, decode_step = "prefill", prompt, 0
    for _ in range(max_steps):
        ar = sub.prepare_inputs(walk, fwd_info, {"text_inputs": [text_inputs]})
        kw = sub.preprocess(walk, ei, [ar])
        out = sub.forward_batched(walk, ei, **kw)["r0"]
        sub.postprocess("r0", fwd_info, out)
        emitted.append(out["new_token"][0])
        if walk == "decode":
            # Match the engine's flag-off accounting: after decode step j
            # (0-based), generated = j + 2 = tokens emitted so far.
            fwd_info.dynamic_loop_iter_counts["decode_loop"] = decode_step
            decode_step += 1
        if sub.check_stop("r0", fwd_info, out):
            break
        walk, text_inputs = "decode", out["text_inputs"][0]
    return torch.cat(emitted)


def _run_pair(k, max_tokens, ignore_eos, seed=0, eos_ids=None):
    """One model, two runs: flag toggled off then on. Same weights by
    construction — toggling beats reseeding because the MTP module's
    parameters would otherwise shift the trunk's RNG draws."""
    torch.manual_seed(seed)
    cfg = _mtp_cfg(k)
    if eos_ids is not None:
        cfg.eos_token_ids = eos_ids
    model = Glm52ForCausalLM(cfg)
    prompt = torch.arange(5, dtype=torch.long) + 3

    streams, handles = [], []
    for mode_k in (0, k):
        cfg.mtp_num_draft_tokens = mode_k
        sub = Glm52LLMSubmodule(model, cfg)
        handle = ReferenceCacheHandle(["r0"])
        streams.append(_drive(
            sub, handle, prompt, _fwd_info(max_tokens, ignore_eos)))
        handles.append(handle)
    cfg.mtp_num_draft_tokens = k
    return streams, handles


def test_mtp_stream_matches_baseline_bitwise():
    (base, spec), _ = _run_pair(k=2, max_tokens=24, ignore_eos=True)
    assert torch.equal(base, spec), (
        f"MTP-on diverged from baseline: {base.tolist()} vs {spec.tolist()}")


def test_mtp_stream_matches_with_k3():
    (base, spec), _ = _run_pair(k=3, max_tokens=17, ignore_eos=True, seed=1)
    assert torch.equal(base, spec)


def test_mtp_stops_exactly_at_max_tokens():
    (base, spec), _ = _run_pair(k=2, max_tokens=7, ignore_eos=True)
    assert base.shape[0] == 7  # engine flag-off accounting
    assert spec.shape[0] == 7  # in-step budget truncation
    assert torch.equal(base, spec)


def test_mtp_eos_truncation_matches_baseline():
    # Data-driven EOS: pick a token the baseline emits mid-stream, declare
    # it a stop id, and rerun both modes — they must stop at exactly its
    # first occurrence, even when it lands inside an accepted draft run.
    (base, _), _ = _run_pair(k=2, max_tokens=24, ignore_eos=True)
    eos = int(base[len(base) // 2])
    (base2, spec2), _ = _run_pair(
        k=2, max_tokens=24, ignore_eos=False, eos_ids=(eos,))
    first = (base2 == eos).nonzero()[0, 0]
    assert base2.shape[0] == int(first) + 1
    assert torch.equal(base2, spec2)


def test_mtp_trunk_replay_seam_bit_identical():
    """THE capture-fix property: routing the decode trunk through the
    piecewise-runner seam (plan -> trunk forward -> cloned outputs, with
    preprocess skipping its own plan) emits the bit-identical stream to
    the fully eager MTP path."""
    torch.manual_seed(0)
    cfg = _mtp_cfg(2)
    model = Glm52ForCausalLM(cfg)
    prompt = torch.arange(5, dtype=torch.long) + 3

    sub_eager = Glm52LLMSubmodule(model, cfg)
    stream_eager = _drive(
        sub_eager, ReferenceCacheHandle(["r0"]), prompt, _fwd_info(24, True))

    sub_replay = Glm52LLMSubmodule(model, cfg)
    handle = ReferenceCacheHandle(["r0"])
    runner = _StubTrunkRunner(sub_replay, handle)
    stream_replay = _drive(
        sub_replay, handle, prompt, _fwd_info(24, True),
        runners={"mtp_trunk": runner})

    assert runner.calls > 0, "decode trunk never went through the runner seam"
    assert torch.equal(stream_eager, stream_replay), (
        f"replay-seam stream diverged: {stream_eager.tolist()} vs "
        f"{stream_replay.tolist()}")


def test_mtp_draft_replay_seam_bit_identical():
    """Routing the chain iterations through the mtp_draft runner seam must
    draft — and therefore emit — the bit-identical stream to the eager
    loop. k=3 gives two chained iterations per step; the stub also pins
    the replay-plan position arithmetic (counter+1 == st+it)."""
    torch.manual_seed(0)
    cfg = _mtp_cfg(3)
    model = Glm52ForCausalLM(cfg)
    prompt = torch.arange(5, dtype=torch.long) + 3

    sub_eager = Glm52LLMSubmodule(model, cfg)
    stream_eager = _drive(
        sub_eager, ReferenceCacheHandle(["r0"]), prompt, _fwd_info(24, True))

    sub_replay = Glm52LLMSubmodule(model, cfg)
    handle = ReferenceCacheHandle(["r0"])
    draft_runner = _StubDraftRunner(sub_replay, handle)
    stream_replay = _drive(
        sub_replay, handle, prompt, _fwd_info(24, True),
        runners={"mtp_draft": draft_runner})

    assert draft_runner.calls > 0, "chain never went through the runner seam"
    assert torch.equal(stream_eager, stream_replay), (
        f"draft-seam stream diverged: {stream_eager.tolist()} vs "
        f"{stream_replay.tolist()}")

    # Both seams together (the production shape at k>=2).
    sub_both = Glm52LLMSubmodule(model, cfg)
    handle_b = ReferenceCacheHandle(["r0"])
    stream_both = _drive(
        sub_both, handle_b, prompt, _fwd_info(24, True),
        runners={
            "mtp_trunk": _StubTrunkRunner(sub_both, handle_b),
            "mtp_draft": _StubDraftRunner(sub_both, handle_b),
        })
    assert torch.equal(stream_eager, stream_both)


def test_mtp_sync_replay_seam_bit_identical():
    """THE padded-sync property, provable on CPU: routing the decode sync
    pass through the padded mtp_sync seam — pads written into the plane
    store and all — emits the bit-identical stream to the eager unpadded
    sync, and the plane still holds exactly counter entries afterwards.
    k=3 exercises e in {1..4} against rows=4 across the run."""
    torch.manual_seed(0)
    cfg = _mtp_cfg(3)
    model = Glm52ForCausalLM(cfg)
    prompt = torch.arange(5, dtype=torch.long) + 3

    sub_eager = Glm52LLMSubmodule(model, cfg)
    h_eager = ReferenceCacheHandle(["r0"])
    stream_eager = _drive(sub_eager, h_eager, prompt, _fwd_info(24, True))

    sub_sync = Glm52LLMSubmodule(model, cfg)
    handle = ReferenceCacheHandle(["r0"])
    sync_runner = _StubSyncRunner(sub_sync, handle)
    stream_sync = _drive(
        sub_sync, handle, prompt, _fwd_info(24, True),
        runners={"mtp_sync": sync_runner})

    assert sync_runner.calls > 0, "decode sync never went through the seam"
    assert torch.equal(stream_eager, stream_sync), (
        f"padded-sync stream diverged: {stream_eager.tolist()} vs "
        f"{stream_sync.tolist()}")
    # Shift-by-one alignment must survive padding: plane entries == counter.
    plane = handle.committed_rows("r0", cfg.num_hidden_layers)
    assert len(plane) == handle._states["r0"].position_id_start

    # All three seams together — the production shape after this lands.
    sub_all = Glm52LLMSubmodule(model, cfg)
    h_all = ReferenceCacheHandle(["r0"])
    stream_all = _drive(
        sub_all, h_all, prompt, _fwd_info(24, True),
        runners={
            "mtp_trunk": _StubTrunkRunner(sub_all, h_all),
            "mtp_draft": _StubDraftRunner(sub_all, h_all),
            "mtp_sync": _StubSyncRunner(sub_all, h_all),
        })
    assert torch.equal(stream_eager, stream_all)


def test_preprocess_skips_plan_only_when_trunk_replays():
    """The replay decision is made once, in preprocess: with a runnable
    trunk graph the eager plan is skipped (the runner's plan is the live
    one) and the runner rides the kwargs; without one, preprocess plans
    as before and the forward gets None."""
    torch.manual_seed(0)
    cfg = _mtp_cfg(2)
    sub = Glm52LLMSubmodule(Glm52ForCausalLM(cfg), cfg)
    fwd = _fwd_info(10, True)
    decode_inputs = {"text_inputs": [torch.tensor([1, 2, 3])]}

    handle = ReferenceCacheHandle(["r0"])
    runner = _StubTrunkRunner(sub, handle)
    draft = _StubDraftRunner(sub, handle)
    sync = _StubSyncRunner(sub, handle)
    ar = sub.prepare_inputs("decode", fwd, decode_inputs)
    kw = sub.preprocess(
        "decode",
        _EngineInputs(handle, _ArgmaxSampler(),
                      {"mtp_trunk": runner, "mtp_draft": draft,
                       "mtp_sync": sync}),
        [ar])
    assert kw["mtp_trunk_runner"] is runner
    assert kw["mtp_draft_runner"] is draft
    assert kw["mtp_sync_runner"] is sync
    assert handle._plan is None, "eager plan should be skipped on replay steps"

    handle2 = ReferenceCacheHandle(["r0"])
    ar2 = sub.prepare_inputs("decode", fwd, decode_inputs)
    kw2 = sub.preprocess(
        "decode", _EngineInputs(handle2, _ArgmaxSampler()), [ar2])
    assert kw2["mtp_trunk_runner"] is None
    assert kw2["mtp_draft_runner"] is None
    assert kw2["mtp_sync_runner"] is None
    assert handle2._plan is not None, "eager path must still plan"

    # Prefill drafts too: the chain runner rides along; the trunk and the
    # padded sync (decode-shape graphs) do not, and the prefill plan is
    # untouched — its sync spans the whole prompt, outside the k+1 family.
    handle3 = ReferenceCacheHandle(["r0"])
    ar3 = sub.prepare_inputs("prefill", fwd, decode_inputs)
    kw3 = sub.preprocess(
        "prefill",
        _EngineInputs(handle3, _ArgmaxSampler(),
                      {"mtp_trunk": runner, "mtp_draft": draft,
                       "mtp_sync": sync}),
        [ar3])
    assert kw3["mtp_trunk_runner"] is None
    assert kw3["mtp_draft_runner"] is draft
    assert kw3["mtp_sync_runner"] is None
    assert handle3._plan is not None, "prefill must keep its eager plan"


def test_mtp_disables_full_forward_capture_configs():
    """With MTP on, no full-forward CUDA-graph configs may register: their
    warmup captures crash (host-side verify/rewind; packed prefill never
    runs preprocess) and the failure mode is a silent 13x eager fallback.
    Flag off keeps the decode + prefill capture pair."""
    cfg = _mtp_cfg(2)
    sub = Glm52LLMSubmodule(Glm52ForCausalLM(cfg), cfg)
    assert sub.get_cuda_graph_configs(torch.device("cpu")) == []
    cfg.mtp_num_draft_tokens = 0
    assert len(sub.get_cuda_graph_configs(torch.device("cpu"))) == 2


def test_mtp_trunk_piecewise_config_shapes():
    """The trunk graph registers exactly one (bs, [k+1]*bs) PACKED bucket
    per capture batch size — never the bs x token-bucket cross product —
    and k >= 2 adds the 1-row-per-request draft-chain graph."""
    from mstar.engine.cuda_graph_config import PiecewiseConfigType

    cfg = _mtp_cfg(2)  # rows per request = 3
    sub = Glm52LLMSubmodule(Glm52ForCausalLM(cfg), cfg)
    configs = sub.get_piecewise_cuda_graph_configs(
        torch.device("cpu"), torch.bfloat16, tp_world_size=1)
    assert set(configs) == {"mtp_trunk", "mtp_draft", "mtp_sync"}
    pc = configs["mtp_trunk"]
    assert pc.get_config_type() == PiecewiseConfigType.PACKED
    assert pc.uses_kv_cache and pc.cache_labels == ["main"]

    shapes = pc.get_capture_shapes(pc.capture_batch_sizes)
    assert [(s.bs, s.total_tokens) for s in shapes] == [
        (1, 3), (2, 6), (4, 12), (8, 24), (16, 48)]
    assert all(s.seq_lens == [3] * s.bs for s in shapes)

    static = pc.make_static_inputs(shapes[0])
    assert set(static) == {"input_ids", "position_ids"}
    assert static["input_ids"].shape == (3,)
    assert static["input_ids"].dtype == torch.long

    dc = configs["mtp_draft"]
    assert dc.get_config_type() == PiecewiseConfigType.PACKED
    assert dc.uses_kv_cache and dc.cache_labels == ["main"]
    dshapes = dc.get_capture_shapes(dc.capture_batch_sizes)
    assert [(s.bs, s.total_tokens) for s in dshapes] == [
        (1, 1), (2, 2), (4, 4), (8, 8), (16, 16)]
    dstatic = dc.make_static_inputs(dshapes[1])
    assert set(dstatic) == {"draft_ids", "prev_hidden", "position_ids"}
    assert dstatic["draft_ids"].shape == (2,)
    assert dstatic["prev_hidden"].shape == (2, cfg.hidden_size)
    assert dstatic["prev_hidden"].dtype == torch.bfloat16

    # The padded sync graph shares the trunk's capture shape exactly.
    sc = configs["mtp_sync"]
    assert sc.get_config_type() == PiecewiseConfigType.PACKED
    sshapes = sc.get_capture_shapes(sc.capture_batch_sizes)
    assert [(s.bs, s.total_tokens) for s in sshapes] == [
        (1, 3), (2, 6), (4, 12), (8, 24), (16, 48)]
    sstatic = sc.make_static_inputs(sshapes[0])
    assert set(sstatic) == {"sync_ids", "pair_hidden", "position_ids"}
    assert sstatic["sync_ids"].shape == (3,)
    assert sstatic["pair_hidden"].shape == (3, cfg.hidden_size)
    assert sstatic["pair_hidden"].dtype == torch.bfloat16

    # k=1 has no chain iterations — no draft graph to pay capture for —
    # but the sync pass (rows=2) still captures.
    cfg.mtp_num_draft_tokens = 1
    assert set(sub.get_piecewise_cuda_graph_configs(
        torch.device("cpu"), torch.bfloat16, tp_world_size=1)) == {
            "mtp_trunk", "mtp_sync"}

    cfg.mtp_num_draft_tokens = 0
    assert sub.get_piecewise_cuda_graph_configs(
        torch.device("cpu"), torch.bfloat16, tp_world_size=1) == {}
    cfg.mtp_num_draft_tokens = 2


def test_piecewise_capture_failure_frees_dummy_kv(monkeypatch):
    """The 2026-08-09 server kill: a failed piecewise capture that keeps its
    dummy pages skews this rank's free-page count, and since capture
    failures land on different ranks at different buckets, the TP symmetry
    guard rejects the allocator state and the whole server dies at warmup.
    _capture_one must free dummy KV state on failure exactly as on
    success."""
    from mstar.engine.cuda_graph_runner import PiecewiseCudaGraphRunner

    cfg = _mtp_cfg(2)
    sub = Glm52LLMSubmodule(Glm52ForCausalLM(cfg), cfg)
    pc = sub.get_piecewise_cuda_graph_configs(
        torch.device("cpu"), torch.bfloat16, tp_world_size=1)["mtp_trunk"]

    runner = PiecewiseCudaGraphRunner.__new__(PiecewiseCudaGraphRunner)
    runner.config = pc
    runner.cache_labels = ["main"]
    freed = []

    class _FakeAlloc:
        def reset_label(self, rid, label, free=False):
            freed.append((rid, label, free))

    runner.alloc_manager = _FakeAlloc()
    rids = ["__pcgr_a__", "__pcgr_b__"]
    monkeypatch.setattr(
        runner, "_setup_cache_manager", lambda shape: (object(), rids))

    def _boom(*args, **kwargs):
        raise RuntimeError("capture blew up mid-graph")

    monkeypatch.setattr(runner, "_capture_one_inner", _boom)

    shape = pc.get_capture_shapes([2])[0]
    try:
        runner._capture_one(shape)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected the capture failure to propagate")
    assert freed == [(rid, "main", True) for rid in rids]


def test_all_captures_failed_logs_error(caplog):
    """Zero-of-N captured must be an ERROR naming the eager consequence;
    partial failure a WARNING; full success silent."""
    import logging as _logging

    from mstar.engine.cuda_graph_runner import _log_capture_outcome

    with caplog.at_level(
        _logging.INFO, logger="mstar.engine.cuda_graph_runner",
    ):
        _log_capture_outcome("LLM", attempted=296, captured=0)
        _log_capture_outcome("LLM", attempted=10, captured=7)
        _log_capture_outcome("LLM", attempted=5, captured=5)
    errors = [r for r in caplog.records if r.levelno == _logging.ERROR]
    warns = [r for r in caplog.records if r.levelno == _logging.WARNING]
    assert len(errors) == 1 and "EAGER" in errors[0].getMessage()
    assert len(warns) == 1 and "3/10" in warns[0].getMessage()


def test_mtp_trunk_kv_and_plane_bookkeeping():
    (base, spec), (h_off, h_on) = _run_pair(k=2, max_tokens=16, ignore_eos=True)
    assert torch.equal(base, spec)
    # Committed trunk KV must trail the baseline's (accepted rows are
    # recomputed identically; rejected tails sit above the counter).
    n = min(h_off._states["r0"].position_id_start,
            h_on._states["r0"].position_id_start)
    for layer in range(4):
        off_rows = h_off.committed_rows("r0", layer)
        on_rows = h_on.committed_rows("r0", layer)
        for p in range(n):
            for c in (0, 1):
                # equal_nan: the unloaded fixture model has uninitialized
                # (torch.empty) params, so NaNs appear — identically in
                # both runs, which is exactly the assertion.
                assert torch.allclose(
                    off_rows[p][c].float(), on_rows[p][c].float(),
                    atol=1e-5, rtol=1e-4, equal_nan=True,
                ), f"layer {layer} pos {p} KV diverged"
    # The MTP plane exists only in the flag-on run, holds exactly counter
    # entries (the shift-by-one alignment), and the baseline never touched
    # layer 4.
    assert ("r0", 4) not in h_off._store
    on_plane = h_on.committed_rows("r0", 4)
    assert len(on_plane) == h_on._states["r0"].position_id_start


def test_mtp_acceptance_log_per_position(caplog):
    """The 512-step acceptance line must carry the conditional per-position
    profile (the datum that separates "first draft mediocre" from "chained
    drafts collapse"). Short tests never cross the threshold, so drive the
    method directly with a synthetic histogram."""
    import logging as _logging

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
    with caplog.at_level(
        _logging.INFO, logger="mstar.model.glm52.submodules",
    ):
        Glm52LLMSubmodule._maybe_log_mtp_acceptance(ns)
    msgs = [r.getMessage() for r in caplog.records]
    assert any("emitted/step" in m for m in msgs)
    (pos_line,) = [m for m in msgs if "by position" in m]
    assert "[256, 128, 64, 64]" in pos_line
    assert "0.50 0.50 0.50" in pos_line
    assert ns._mtp_stat_logged == 512
    # The line must name which trunk-pairing arm produced it. An acceptance
    # profile whose arm you infer from launch env is one you cannot trust
    # after the fact — and mislabelling an arm silently inverts the A/B.
    assert "pre-final-norm" in pos_line and "POST" not in pos_line

    caplog.clear()
    with caplog.at_level(
        _logging.INFO, logger="mstar.model.glm52.submodules",
    ):
        Glm52LLMSubmodule._maybe_log_mtp_acceptance(_ns(True))
    (post_line,) = [
        r.getMessage() for r in caplog.records if "by position" in r.getMessage()
    ]
    assert "POST-final-norm" in post_line
