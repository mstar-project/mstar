"""Picking and gating the fused TP all-reduce + residual + RMSNorm.

The kernels need several GPUs (see ``test/rocm/test_ar_fusion_tp.py``). These
tests cover the decisions about which code path a rank takes, which have to
come out the same on every rank or the group hangs on the next collective.
That's the platform's strategy, AITER's one-stage gate, the startup refusals,
and how a capture with registered AITER inputs gets finished across ranks.
AITER and the peer ranks are fakes. A test passes the peers' side of each
agreement to the fake ``_agree``.
"""

from __future__ import annotations

import inspect
import sys
import types

import pytest
import torch

from mstar import platforms
from mstar.distributed import ar_fusion
from mstar.distributed.ar_fusion import (
    AiterARFusion,
    ARFusion,
    FlashInferARFusion,
    GraphCaptureScope,
)
from mstar.distributed.communication import CommGroup
from mstar.platforms import CpuPlatform, CudaPlatform, RocmPlatform, XpuPlatform

BF16 = torch.bfloat16


@pytest.fixture(autouse=True)
def fresh_registry(monkeypatch):
    monkeypatch.setattr(ar_fusion, "_FUSIONS", [])
    ar_fusion._create.reset()
    yield
    ar_fusion._create.reset()


@pytest.fixture
def on_platform(monkeypatch):
    def use(platform):
        monkeypatch.setattr(platforms, "resolve_platform", lambda: platform)
        ar_fusion._create.reset()
        return platform

    return use


# --- which strategy -----------------------------------------------------------


class _Tagged(ARFusion):
    def __init__(self, tag):
        self.tag = tag


@pytest.mark.parametrize(
    ("platform", "expected"),
    [(CudaPlatform(), "flashinfer"), (RocmPlatform(), "aiter"),
     (XpuPlatform(), "flashinfer"), (CpuPlatform(), "flashinfer")],
    ids=repr,
)
def test_each_platform_builds_its_strategy(monkeypatch, on_platform, platform, expected):
    """NVIDIA (and anything that is not ROCm) keeps today's FlashInfer path."""
    on_platform(platform)
    monkeypatch.setattr(FlashInferARFusion, "create",
                        classmethod(lambda cls, *a: _Tagged("flashinfer")))
    monkeypatch.setattr(AiterARFusion, "create",
                        classmethod(lambda cls, *a: _Tagged("aiter")))

    first = ar_fusion.create_fusion(0, 2, None, 2048, 4096, BF16)
    second = ar_fusion.create_fusion(0, 2, None, 2048, 4096, BF16)

    assert (first.tag, second.tag) == (expected, expected)
    assert (first.ws_id, second.ws_id) == (0, 1)
    assert ar_fusion._FUSIONS == [first, second]


def test_a_strategy_that_cannot_build_is_not_registered(monkeypatch, on_platform):
    on_platform(RocmPlatform())
    monkeypatch.setattr(AiterARFusion, "create", classmethod(lambda cls, *a: None))
    assert ar_fusion.create_fusion(0, 2, None, 2048, 4096, BF16) is None
    assert ar_fusion._FUSIONS == []


# --- AITER's one-stage gate ---------------------------------------------------


def _vllm_uses_one_stage(world, hidden, tokens, dtype=BF16) -> bool:
    """vLLM's ``AiterCustomAllreduce.use_1stage_fused_ar_rms`` (v0.32.0) on a
    fully connected node, copied over as the reference."""
    if tokens > 80 or dtype not in (torch.bfloat16, torch.float16):
        return False
    elt = torch.empty((), dtype=dtype).element_size()
    pack = 16 // elt
    if hidden % pack or hidden // pack > 1024:
        return False
    if world == 2:
        return True
    nbytes = tokens * hidden * elt
    if world <= 4:
        return nbytes < 256 * 1024
    if world <= 8:
        return nbytes < 128 * 1024
    return False


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("hidden", [512, 2048, 4096, 5120, 8192, 9216, 4100])
@pytest.mark.parametrize("dtype", [BF16, torch.float16])
def test_the_fused_bound_is_vllms_one_stage_predicate(world, hidden, dtype):
    """vLLM's rule is monotone in tokens, so it comes down to a max."""
    bound = AiterARFusion.fused_token_bound(world, hidden, dtype, max_tokens=4096)
    for tokens in range(1, 130):
        assert (tokens <= bound) == _vllm_uses_one_stage(world, hidden, tokens, dtype), (
            world, hidden, tokens,
        )


@pytest.mark.parametrize(
    ("world", "hidden", "bound"),
    [(2, 4096, 80), (2, 5120, 80), (4, 4096, 31), (4, 5120, 25),
     (8, 5120, 12), (8, 8192, 7), (2, 8192, 80)],
)
def test_fused_bounds_at_the_model_shapes(world, hidden, bound):
    assert AiterARFusion.fused_token_bound(world, hidden, BF16, 2048) == bound


@pytest.mark.parametrize(
    ("world", "hidden", "dtype"),
    [(1, 4096, BF16), (6, 4096, BF16), (2, 9216, BF16), (2, 4100, BF16),
     (2, 4096, torch.float32)],
)
def test_shapes_the_one_stage_kernel_cannot_take_build_nothing(world, hidden, dtype):
    assert AiterARFusion.fused_token_bound(world, hidden, dtype, 2048) == 0


def test_the_callers_token_limit_still_applies():
    assert AiterARFusion.fused_token_bound(2, 4096, BF16, max_tokens=16) == 16


def _fusion(**kw) -> AiterARFusion:
    args = dict(ca=None, cpu_group=None, hidden=64, dtype=BF16, max_tokens=8,
                register_in_capture=True)
    args.update(kw)
    return AiterARFusion(**args)


def test_accepts_only_what_the_kernel_was_sized_for():
    f = _fusion()
    x = torch.zeros(8, 64, dtype=BF16)
    assert f.accepts(x, x.clone(), 0.0)
    assert f.accepts(x, x.clone(), 1.0)
    assert not f.accepts(torch.zeros(9, 64, dtype=BF16), torch.zeros(9, 64, dtype=BF16), 0.0)
    assert not f.accepts(torch.zeros(0, 64, dtype=BF16), torch.zeros(0, 64, dtype=BF16), 0.0)
    assert not f.accepts(torch.zeros(8, 32, dtype=BF16), torch.zeros(8, 32, dtype=BF16), 0.0)
    assert not f.accepts(x.half(), x.half(), 0.0), "the workspace's dtype only"
    assert not f.accepts(x, x.float(), 0.0), "AITER reads the residual as x's dtype"
    assert not f.accepts(x, x.clone(), 0.5), "the kernel scales by w or 1 + w"
    assert not f.accepts(torch.zeros(64, 8, dtype=BF16).t(), x, 0.0)
    assert not f.accepts(x, torch.zeros(64, 8, dtype=BF16).t(), 0.0)
    assert not f.accepts(x.view(2, 4, 64), x.view(2, 4, 64), 0.0)


def test_flashinfer_accepts_what_it_did_before():
    f = FlashInferARFusion(workspace=None, max_tokens=4)
    x = torch.zeros(4, 16)
    assert f.accepts(x, x, 0.5)
    assert not f.accepts(torch.zeros(5, 16), torch.zeros(5, 16), 0.0)
    assert not f.accepts(torch.zeros(16, 4).t(), x, 0.0)


# --- a fake AITER -------------------------------------------------------------


class _FakeCA:
    """``CustomAllreduce`` as far as the strategy uses it."""

    built: list[_FakeCA] = []

    def __init__(self, group, device, max_size=1 << 30):
        self.group, self.device, self.max_size = group, device, max_size
        self.disabled = False
        self.enable_register_for_capturing = True
        self._ptr = 7
        self.pending = 0
        self.flushes = 0
        self.closed = False
        self.calls: list[dict] = []
        _FakeCA.built.append(self)

    def _ops_get_graph_buffer_count(self, ptr):
        assert ptr == self._ptr
        return self.pending

    def register_graph_buffers(self):
        self.flushes += 1
        self.pending = 0

    def fused_ar_rms(self, inp, res_inp, *, res_out=None, out=None, scale_out=None,
                     w, eps, registered=False, use_1stage=False,
                     post_per_token_quant=False, out_hidden_dim=0, gemma_norm=False,
                     emit_bf16=False):
        self.calls.append(dict(registered=registered, use_1stage=use_1stage,
                               gemma_norm=gemma_norm, w_dtype=w.dtype, eps=eps))
        r = inp + res_inp
        v = r.float() * torch.rsqrt(r.float().pow(2).mean(-1, keepdim=True) + eps)
        normed = (v * (w.float() + float(gemma_norm))).to(inp.dtype)
        if out is not None:
            out.copy_(normed)
            res_out.copy_(r)
            return out, res_out
        return normed, r

    def close(self):
        self.closed = True


class _FakePool:
    _pool_seq = 0


@pytest.fixture
def fake_aiter(monkeypatch):
    """Installs the fake as ``aiter``. The test sets the version it reports."""
    _FakeCA.built = []
    _FakePool._pool_seq = 0
    mod = types.ModuleType("aiter.dist.device_communicators.custom_all_reduce")
    mod.CustomAllreduce = _FakeCA
    mod.IPCBufferPool = _FakePool
    for name in ("aiter", "aiter.dist", "aiter.dist.device_communicators"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    state = types.SimpleNamespace(version="0.1.25+rocm7.2.manylinux.2.28")
    monkeypatch.setattr(ar_fusion, "_aiter_version", lambda: state.version)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    return state


class _Peers:
    """Fake ``_agree``, the max of this rank's values and the peers' next ones."""

    def __init__(self, *rounds: list[int]):
        self.rounds = list(rounds)
        self.seen: list[list[int]] = []

    def __call__(self, cpu_group, values):
        self.seen.append(list(values))
        peer = self.rounds.pop(0) if self.rounds else [0] * len(values)
        return [max(a, b) for a, b in zip(values, peer, strict=True)]


def _create(monkeypatch, peers=None, check=None, world=2, hidden=4096):
    peers = peers or _Peers()
    monkeypatch.setattr(ar_fusion, "_agree", peers)
    monkeypatch.setattr(AiterARFusion, "_check_kernel", staticmethod(lambda *a: check))
    fusion = AiterARFusion.create(0, world, "gloo-group", 2048, hidden, BF16)
    return fusion, peers


def test_builds_a_workspace_sized_to_the_fused_bound(fake_aiter, monkeypatch):
    fusion, peers = _create(monkeypatch, world=4, hidden=5120)

    (ca,) = _FakeCA.built
    assert ca.group == "gloo-group", "AITER needs a non-NCCL group"
    # 25 tokens x 5120 x 2 B = 250 KiB, rounded up to the 1 MiB floor
    assert ca.max_size == 1 << 20
    assert fusion.max_tokens == 25 and fusion.hidden == 5120
    assert len(peers.seen) == 3, "refusal, build and startup check are each agreed"


def test_a_large_bound_rounds_max_size_up_to_whole_mebibytes(fake_aiter, monkeypatch):
    _create(monkeypatch, world=2, hidden=8192)  # 80 x 8192 x 2 B = 1.25 MiB
    assert _FakeCA.built[0].max_size == 2 << 20


def test_pool_keys_start_from_the_groups_highest_count(fake_aiter, monkeypatch):
    """AITER names its store keys after a per-process pool counter. A member
    that already built a pool for another group would otherwise wait on keys
    nobody writes."""
    _FakePool._pool_seq = 1
    peers = _Peers([0, 3])
    _create(monkeypatch, peers=peers)
    assert peers.seen[0] == [0, 1]
    assert _FakePool._pool_seq == 3


@pytest.mark.parametrize(
    ("version", "builds"),
    [("0.1.19", False), ("0.1.19.post1", True), ("0.1.21.post2", True),
     ("0.1.25+rocm7.2.manylinux.2.28", True), ("0.1.25+rocm7.2.manylinux_2_28", True),
     ("garbage", False), (None, False)],
)
def test_aiter_releases_without_the_one_stage_exit_barrier_are_refused(
    fake_aiter, monkeypatch, version, builds,
):
    fake_aiter.version = version
    fusion, peers = _create(monkeypatch)
    assert (fusion is not None) == builds
    assert bool(_FakeCA.built) == builds, "a refusal builds nothing"
    assert peers.seen[0][0] == int(not builds), "and is agreed before any build"


def test_aiter_without_gemma_norm_is_refused(fake_aiter, monkeypatch):
    def old_signature(self, inp, res_inp, *, w, eps, registered=False, use_1stage=False):
        raise AssertionError("never called")

    monkeypatch.setattr(_FakeCA, "fused_ar_rms", old_signature)
    assert "gemma_norm" not in inspect.signature(_FakeCA.fused_ar_rms).parameters
    fusion, _ = _create(monkeypatch)
    assert fusion is None and _FakeCA.built == []


def test_a_peers_refusal_keeps_this_rank_out(fake_aiter, monkeypatch):
    fusion, _ = _create(monkeypatch, peers=_Peers([1, 0]))
    assert fusion is None and _FakeCA.built == []


def test_a_failed_construction_anywhere_is_a_failure_everywhere(fake_aiter, monkeypatch):
    fusion, _ = _create(monkeypatch, peers=_Peers([0, 0], [1]))
    assert fusion is None
    (ca,) = _FakeCA.built
    assert ca.closed


def test_a_disabled_custom_all_reduce_builds_nothing(fake_aiter, monkeypatch):
    real_init = _FakeCA.__init__

    def disabled(self, *a, **kw):
        real_init(self, *a, **kw)
        self.disabled = True

    monkeypatch.setattr(_FakeCA, "__init__", disabled)
    fusion, peers = _create(monkeypatch)
    assert fusion is None and _FakeCA.built[0].closed
    assert peers.seen[1] == [1]


def test_a_failed_kernel_check_on_a_peer_builds_nothing(fake_aiter, monkeypatch):
    fusion, _ = _create(monkeypatch, peers=_Peers([0, 0], [0], [1, 0]))
    assert fusion is None and _FakeCA.built[0].closed


def test_a_failed_kernel_check_here_builds_nothing(fake_aiter, monkeypatch):
    fusion, peers = _create(monkeypatch, check="norm at tokens=1: max |err| 3")
    assert fusion is None and peers.seen[2][0] == 1


def test_unsupported_shapes_skip_aiter_and_every_collective(fake_aiter, monkeypatch):
    fusion, peers = _create(monkeypatch, world=6)
    assert fusion is None and peers.seen == [] and _FakeCA.built == []


def test_registered_capture_is_agreed(fake_aiter, monkeypatch):
    """Off if any peer's allocator uses expandable segments (AITER then copies
    into its pool during capture). The capture scope's collectives must run on
    every rank or none."""
    fusion, _ = _create(monkeypatch, peers=_Peers([0, 0], [0], [0, 1]))
    assert fusion is not None
    assert type(fusion.graph_capture()) is GraphCaptureScope


# --- the startup kernel check -------------------------------------------------


def test_the_kernel_check_passes_a_correct_kernel():
    ca = _FakeCA(None, "cpu")
    assert AiterARFusion._check_kernel(ca, 0, 1, 64, BF16, 8, torch.device("cpu")) is None
    assert {c["gemma_norm"] for c in ca.calls} == {False, True}
    assert all(c["use_1stage"] and not c["registered"] for c in ca.calls)


def test_the_kernel_check_catches_a_kernel_that_skips_the_hidden_size():
    ca = _FakeCA(None, "cpu")
    ca.fused_ar_rms = lambda inp, res, **kw: (kw["out"], kw["res_out"])
    # the outputs keep the NaN they were filled with
    assert "max |err| inf" in AiterARFusion._check_kernel(ca, 0, 1, 64, BF16, 8, torch.device("cpu"))


def test_the_kernel_check_allows_truncation_at_eight_ranks():
    """AITER truncates to bf16 on gfx942. With eight ranks a correct kernel can
    be off by about one ulp of the sum from a round-to-nearest reference, even
    where the residual nearly cancels the sum."""
    ca = _FakeCA(None, "cpu")

    def truncating(inp, res, *, res_out, out, w, eps, gemma_norm, **kw):
        gen = torch.Generator().manual_seed(1234)
        tokens, hidden = inp.shape
        xs = torch.randn(8, tokens, hidden, generator=gen).to(BF16)
        a = _truncate(xs.float().sum(0))
        r = _truncate(a.float() + res.float())
        v = r.float()
        v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
        out.copy_(_truncate(v * (w.float() + float(gemma_norm))))
        res_out.copy_(r)
        return out, res_out

    ca.fused_ar_rms = truncating
    assert AiterARFusion._check_kernel(ca, 0, 8, 64, BF16, 1, torch.device("cpu")) is None


def _truncate(v: torch.Tensor) -> torch.Tensor:
    """fp32 -> bf16 by dropping the low 16 bits, as AITER does on gfx942."""
    bits = v.float().contiguous().view(torch.int32) & ~0xFFFF
    return bits.view(torch.float32).to(BF16)


def test_the_kernel_check_catches_a_kernel_that_ignores_gemma():
    ca = _FakeCA(None, "cpu")
    real = ca.fused_ar_rms
    ca.fused_ar_rms = lambda inp, res, **kw: real(inp, res, **{**kw, "gemma_norm": False})
    assert "gemma=True" in AiterARFusion._check_kernel(ca, 0, 1, 64, BF16, 8, torch.device("cpu"))


def test_the_kernel_check_reports_a_raising_kernel():
    ca = _FakeCA(None, "cpu")

    def boom(*a, **kw):
        raise RuntimeError("no kernel for this size")

    ca.fused_ar_rms = boom
    assert "no kernel" in AiterARFusion._check_kernel(ca, 0, 1, 64, BF16, 8, torch.device("cpu"))


# --- running and capturing ----------------------------------------------------


def test_run_is_the_one_stage_kernel_in_the_activation_dtype():
    ca = _FakeCA(None, "cpu")
    f = _fusion(ca=ca)
    x = torch.randn(4, 64).to(BF16)
    f.run(x, x.clone(), torch.ones(64), 1e-5, 1.0)
    (call,) = ca.calls
    assert call["use_1stage"] and call["gemma_norm"] and call["w_dtype"] == BF16
    assert call["eps"] == 1e-5 and not call["registered"]


def test_inputs_are_registered_only_while_a_scope_is_capturing(monkeypatch):
    ca = _FakeCA(None, "cpu")
    f = _fusion(ca=ca)
    monkeypatch.setattr(ar_fusion, "_agree", _Peers())
    x = torch.randn(4, 64).to(BF16)
    capturing = types.SimpleNamespace(on=False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing.on)

    with f.graph_capture() as scope:
        f.run(x, x, torch.ones(64, dtype=BF16), 1e-6, 0.0)  # warmup
        capturing.on = True
        f.run(x, x, torch.ones(64, dtype=BF16), 1e-6, 0.0)  # capture
        scope.captured = True
    f.run(x, x, torch.ones(64, dtype=BF16), 1e-6, 0.0)  # replay-time eager

    assert [c["registered"] for c in ca.calls] == [False, True, False]


def _finish(monkeypatch, *, pending, peer, captured=True):
    ca = _FakeCA(None, "cpu")
    ca.pending = pending
    f = _fusion(ca=ca)
    peers = _Peers(peer)
    monkeypatch.setattr(ar_fusion, "_agree", peers)
    with f.graph_capture() as scope:
        scope.captured = captured
    return f, ca, scope, peers


def test_a_capture_recorded_alike_everywhere_is_registered(monkeypatch):
    f, ca, scope, peers = _finish(monkeypatch, pending=3, peer=[3, -3, 0])
    assert scope.usable and ca.flushes == 1
    assert peers.seen == [[3, -3, 0]]
    assert isinstance(f.graph_capture(), ar_fusion._RegisteredCapture)


def test_a_capture_with_nothing_recorded_needs_no_registration(monkeypatch):
    _, ca, scope, _ = _finish(monkeypatch, pending=0, peer=[0, 0, 0])
    assert scope.usable and ca.flushes == 0


def test_differing_counts_drop_the_graph_and_stop_registering(monkeypatch):
    """Registration pairs up the ranks' i-th addresses, so with different counts
    the pairs would be wrong. Nothing gets registered and the graph is dropped.
    Later captures copy into the pool, which doesn't need pairing."""
    f, ca, scope, _ = _finish(monkeypatch, pending=3, peer=[4, -3, 0])
    assert not scope.usable and ca.flushes == 0
    assert type(f.graph_capture()) is GraphCaptureScope


def test_a_peers_failed_capture_drops_the_graph_here(monkeypatch):
    f, ca, scope, _ = _finish(monkeypatch, pending=2, peer=[2, -2, 1])
    assert not scope.usable
    assert ca.flushes == 1, "equal counts still pair up, registering keeps ranks in step"
    assert isinstance(f.graph_capture(), ar_fusion._RegisteredCapture)


def test_a_failed_capture_here_is_reported(monkeypatch):
    _, _, scope, peers = _finish(monkeypatch, pending=0, peer=[0, 0, 0], captured=False)
    assert not scope.usable and peers.seen == [[0, 0, 1]]


def test_unwinding_through_the_scope_runs_no_collective(monkeypatch):
    ca = _FakeCA(None, "cpu")
    f = _fusion(ca=ca)
    peers = _Peers()
    monkeypatch.setattr(ar_fusion, "_agree", peers)
    with pytest.raises(KeyboardInterrupt):
        with f.graph_capture():
            raise KeyboardInterrupt
    assert peers.seen == [] and not f._capturing


def test_the_plain_scope_keeps_what_was_captured():
    with GraphCaptureScope() as scope:
        scope.captured = True
    assert scope.usable
    with GraphCaptureScope() as scope:
        pass
    assert not scope.usable


# --- CommGroup ----------------------------------------------------------------


class _Recording(ARFusion):
    def __init__(self, accept: bool):
        self.accept = accept
        self.runs = 0

    def accepts(self, x, residual, weight_bias):
        return self.accept

    def run(self, x, residual, weight, eps, weight_bias):
        self.runs += 1
        return x * 0 + 7, residual * 0 + 9


def _group_with(monkeypatch, fusion) -> CommGroup:
    group = CommGroup(0, 0, [0, 1])
    if fusion is not None:
        fusion.ws_id = len(ar_fusion._FUSIONS)
        ar_fusion._FUSIONS.append(fusion)
    group._ar_fusion = fusion
    # stands in for the peer's zero contribution
    monkeypatch.setattr(group, "all_reduce", lambda t: t)
    return group


def test_accepted_inputs_go_through_the_fused_op(monkeypatch):
    fusion = _Recording(accept=True)
    group = _group_with(monkeypatch, fusion)
    x = torch.randn(3, 16)
    normed, r = group.allreduce_add_rmsnorm(x, x.clone(), torch.ones(16), 1e-6)
    assert fusion.runs == 1
    assert torch.equal(normed, torch.full_like(x, 7)) and torch.equal(r, torch.full_like(x, 9))


@pytest.mark.parametrize("weight_bias", [0.0, 1.0])
def test_the_rest_keep_the_collective_and_a_separate_norm(monkeypatch, weight_bias):
    fusion = _Recording(accept=False)
    group = _group_with(monkeypatch, fusion)
    x, res, w = torch.randn(3, 16), torch.randn(3, 16), torch.randn(16)

    normed, r = group.allreduce_add_rmsnorm(x, res, w, 1e-6, weight_bias)

    assert fusion.runs == 0
    want_r = x + res
    want = want_r * torch.rsqrt(want_r.pow(2).mean(-1, keepdim=True) + 1e-6) * (w + weight_bias)
    torch.testing.assert_close(r, want_r)
    torch.testing.assert_close(normed, want)


def test_without_a_fusion_capture_scopes_are_plain(monkeypatch):
    group = _group_with(monkeypatch, None)
    assert type(group.graph_capture()) is GraphCaptureScope


def test_single_rank_groups_never_build_a_fusion(monkeypatch):
    built = []
    monkeypatch.setattr(ar_fusion, "create_fusion", lambda *a: built.append(a))
    assert CommGroup.trivial().init_allreduce_fusion(4096, BF16) is False
    assert built == []


def test_init_builds_once_per_group(monkeypatch):
    calls = []

    def create(*args):
        calls.append(args)
        return _Recording(accept=True)

    monkeypatch.setattr(ar_fusion, "create_fusion", create)
    group = CommGroup(1, 1, [0, 1])
    group.cpu_group = "gloo"
    assert group.init_allreduce_fusion(4096, BF16, max_tokens=64)
    assert group.init_allreduce_fusion(4096, BF16, max_tokens=64)
    assert calls == [(1, 2, "gloo", 64, 4096, BF16)]
