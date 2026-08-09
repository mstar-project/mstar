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

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Same import-time stubs as test_glm52_mtp.py — construction-level tests on
# machines without CUDA wheels.
if "flashinfer" not in sys.modules:
    def _cpu_rmsnorm(x, weight, eps=1e-6):
        x32 = x.float()
        normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
        return (normed * weight.float()).to(x.dtype)

    _fi = types.ModuleType("flashinfer")
    _fi.norm = types.SimpleNamespace(rmsnorm=_cpu_rmsnorm)
    sys.modules["flashinfer"] = _fi

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
        hidden = self.sub._hidden(
            static_inputs["input_ids"], static_inputs["position_ids"], self.handle)
        return {"hidden": hidden.clone(), "logits": self.sub.lm_head(hidden).clone()}


def _mtp_cfg(k: int) -> Glm52ModelConfig:
    cfg = Glm52ModelConfig.reduced()
    cfg.num_hidden_layers = 4  # MTP position lands FULL (4 = offset-1 + freq)
    cfg.mtp_num_draft_tokens = k
    return cfg


def _fwd_info(max_tokens: int, ignore_eos: bool) -> SimpleNamespace:
    return SimpleNamespace(
        request_id="r0",
        max_tokens=max_tokens,
        sampling_config={"LLM": SimpleNamespace(ignore_eos=ignore_eos)},
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
    ar = sub.prepare_inputs("decode", fwd, decode_inputs)
    kw = sub.preprocess(
        "decode", _EngineInputs(handle, _ArgmaxSampler(), {"mtp_trunk": runner}),
        [ar])
    assert kw["mtp_trunk_runner"] is runner
    assert handle._plan is None, "eager plan should be skipped on replay steps"

    handle2 = ReferenceCacheHandle(["r0"])
    ar2 = sub.prepare_inputs("decode", fwd, decode_inputs)
    kw2 = sub.preprocess(
        "decode", _EngineInputs(handle2, _ArgmaxSampler()), [ar2])
    assert kw2["mtp_trunk_runner"] is None
    assert handle2._plan is not None, "eager path must still plan"


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
    per capture batch size — never the bs x token-bucket cross product."""
    from mstar.engine.cuda_graph_config import PiecewiseConfigType

    cfg = _mtp_cfg(2)  # rows per request = 3
    sub = Glm52LLMSubmodule(Glm52ForCausalLM(cfg), cfg)
    configs = sub.get_piecewise_cuda_graph_configs(
        torch.device("cpu"), torch.bfloat16, tp_world_size=1)
    assert set(configs) == {"mtp_trunk"}
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

    cfg.mtp_num_draft_tokens = 0
    assert sub.get_piecewise_cuda_graph_configs(
        torch.device("cpu"), torch.bfloat16, tp_world_size=1) == {}


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
