"""GLM's routed experts on flashinfer's Blackwell fp8 MoE (fused_moe.trtllm_fp8): the layout transforms are exact
permutations; the repacked experts, read the way the kernel documents its layout, compute what the block's own
experts do; the probe gates the kernel per device and shape, and a failed or raising probe keeps Triton; a block
that took the kernel runs it at every token count. On sm_100 / sm_103 only: the kernel against the Triton kernels
and a dequantized reference over several shapes and routings, from a CUDA graph too, and the block end to end."""

import logging
import types

import pytest
import torch
import torch.nn.functional as F

import mstar.model.glm52.components.moe as moe_mod
from benchmark.kernels.bw_moe_trials import REFERENCE_TOLERANCE, dequantized, glm_router, random_experts, reference
from mstar.model.glm52.components.moe import Glm52SparseMoeBlock
from mstar.model.glm52.config import Glm52ModelConfig
from mstar.model.glm52.weight_loader import restore_fp32_params
from mstar.utils.fused_moe import trtllm_fp8

BLACKWELL = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


@pytest.fixture(autouse=True)
def _fresh_verdicts(monkeypatch):
    monkeypatch.setattr(trtllm_fp8, "_VERDICTS", {})
    monkeypatch.setattr(trtllm_fp8, "_LOGGED", set())


def _rows(m: int) -> torch.Tensor:
    """A fixed row permutation standing in for flashinfer's shuffle (any permutation must round-trip)."""
    return torch.randperm(m, generator=torch.Generator().manual_seed(m))


def test_swap_halves_swaps_gate_and_up_and_is_its_own_inverse():
    w = torch.randint(0, 256, (3, 8, 5), dtype=torch.uint8)
    s = torch.rand(3, 4, 2)
    for x in (w, s):
        half = x.shape[1] // 2
        swapped = trtllm_fp8.swap_halves(x)
        assert torch.equal(swapped[:, :half], x[:, half:]) and torch.equal(swapped[:, half:], x[:, :half])
        assert torch.equal(trtllm_fp8.swap_halves(swapped), x)


def test_block_major_k_is_an_exact_permutation():
    w = torch.randint(0, 256, (2, 64, 384), dtype=torch.uint8)
    rows = _rows(64)
    b = trtllm_fp8.to_block_major_k(w, rows)
    assert b.shape == (2, 3, 64, 128) and b.is_contiguous()
    # row r of K block kb holds original row rows[r], columns kb*128 .. kb*128+127
    assert torch.equal(b[1, 2, 5], w[1, rows[5], 256:384])
    assert torch.equal(trtllm_fp8.from_block_major_k(b, rows), w)


def _as_the_kernel_reads(b: torch.Tensor, rows_of) -> torch.Tensor:
    """The kernel's view of a shuffled BlockMajorK weight, as (E, M, K): slot r holds row rows[r]."""
    e, kb, m, block = b.shape
    w = torch.empty_like(b)
    w[:, :, rows_of(m), :] = b
    return w.permute(0, 2, 1, 3).reshape(e, m, kb * block)


def test_pack_puts_up_before_gate_and_keeps_every_byte():
    raw = random_experts(4, 256, 128, "cpu")
    packed = trtllm_fp8.pack(raw, rows_of=_rows)
    assert packed.w13.shape == (4, 2, 256, 128) and packed.w2.shape == (4, 1, 256, 128)
    w13 = _as_the_kernel_reads(packed.w13, _rows)
    assert torch.equal(w13[:, :128], raw.w13[:, 128:]) and torch.equal(w13[:, 128:], raw.w13[:, :128])
    assert torch.equal(packed.s13[:, :1], raw.s13[:, 1:]) and torch.equal(packed.s13[:, 1:], raw.s13[:, :1])
    assert torch.equal(_as_the_kernel_reads(packed.w2, _rows), raw.w2)
    assert packed.s2 is raw.s2


def test_flashinfer_shuffle_rows_round_trip():
    pytest.importorskip("flashinfer")
    for m in (256, 512, 6144):
        rows = trtllm_fp8.shuffle_rows(m)
        assert torch.equal(rows.sort().values, torch.arange(m))
        assert ((rows // 16) == (torch.arange(m) // 16)).all(), "the shuffle keeps rows in their 16-row group"
    w, rows = torch.randint(0, 256, (2, 256, 256), dtype=torch.uint8), trtllm_fp8.shuffle_rows(256)
    assert torch.equal(trtllm_fp8.from_block_major_k(trtllm_fp8.to_block_major_k(w, rows), rows), w)


def test_off_blackwell_the_kernel_is_not_taken():
    raw = random_experts(4, 256, 128, "cpu")
    assert trtllm_fp8.take(raw, 2, (128, 128)) is None
    assert "sm_100" in trtllm_fp8._VERDICTS[(torch.device("cpu"), (4, 256, 256), (4, 256, 128), 2)]


@pytest.fixture
def blackwell_stand_in(monkeypatch):
    """take() as on a Blackwell GPU whose probe passes; ``probes`` counts the probe's runs."""
    probes = []
    monkeypatch.setattr(trtllm_fp8, "unavailable", lambda *a: None)
    monkeypatch.setattr(trtllm_fp8, "shuffle_rows", _rows)
    monkeypatch.setattr(trtllm_fp8, "_probe", lambda raw, packed, top_k: probes.append(top_k))
    monkeypatch.setattr(trtllm_fp8, "autotune", lambda *a: "not tuned (CPU)")
    return probes


def test_the_probe_runs_once_per_device_and_shape(blackwell_stand_in):
    raw = random_experts(4, 256, 128, "cpu")
    first, second = trtllm_fp8.take(raw, 2, (128, 128)), trtllm_fp8.take(raw, 2, (128, 128))
    assert first is not None and second is not None and torch.equal(first.w13, second.w13)
    assert blackwell_stand_in == [2]
    assert trtllm_fp8.take(raw, 3, (128, 128)) is not None
    assert blackwell_stand_in == [2, 3]


@pytest.mark.parametrize("fails", [False, True])
def test_autotune_profiles_the_decode_counts_once_per_shape(monkeypatch, caplog, fails):
    """One call at the largest count inside flashinfer.autotune, with the tuning buckets; a second block of the
    shape reuses the result; a failing tuner logs and keeps the default tactics."""
    import contextlib
    import sys

    seen, calls = [], []

    @contextlib.contextmanager
    def fake_autotune(tune_mode, cache=None, tuning_buckets=None):
        seen.append((tune_mode, cache, tuning_buckets))
        if fails:
            raise RuntimeError("no tactic profiled")
        yield

    monkeypatch.setitem(sys.modules, "flashinfer.autotuner", types.SimpleNamespace(autotune=fake_autotune))
    monkeypatch.setattr(trtllm_fp8, "_TUNED", {})
    monkeypatch.setattr(trtllm_fp8, "experts", lambda x, packed, w, ids: calls.append(tuple(x.shape)) or x)
    packed = trtllm_fp8.ExpertWeights(torch.zeros(4, 2, 512, 128), None, None, torch.zeros(4, 2, 2))
    with caplog.at_level(logging.WARNING, logger=trtllm_fp8.__name__):
        first = trtllm_fp8.autotune(packed, 256, 2, torch.device("cpu"))
        assert trtllm_fp8.autotune(packed, 256, 2, torch.device("cpu")) == first
    assert seen == [(True, None, trtllm_fp8.TUNE_TOKENS)]
    if fails:
        assert first.startswith("autotune failed") and "no tactic profiled" in caplog.text and calls == []
    else:
        assert first.startswith("tactics autotuned at 7 token counts 1..64") and calls == [(64, 256)]


@pytest.mark.parametrize("outcome", ["mismatch", "raises"])
def test_a_failed_probe_keeps_triton_for_the_shape(blackwell_stand_in, monkeypatch, caplog, outcome):
    calls = []

    def probe(raw, packed, top_k):
        calls.append(top_k)
        if outcome == "raises":
            raise RuntimeError("no cubin for this tactic")
        return "relative error 1.41 at 8 tokens, above 0.05"

    monkeypatch.setattr(trtllm_fp8, "_probe", probe)
    raw = random_experts(4, 256, 128, "cpu")
    with caplog.at_level(logging.WARNING, logger=trtllm_fp8.__name__):
        assert trtllm_fp8.take(raw, 2, (128, 128)) is None
        assert trtllm_fp8.take(raw, 2, (128, 128)) is None
    assert calls == [2]
    assert "keep the Triton kernels" in caplog.text
    assert ("no cubin" if outcome == "raises" else "1.41") in caplog.text


def test_the_probe_accepts_rounding_and_refuses_a_wrong_answer(monkeypatch):
    """_probe itself, with the Triton kernels and the kernel replaced by the dequantized reference."""
    from mstar.utils.fused_moe import prefill

    raw = random_experts(4, 256, 128, "cpu")
    packed = trtllm_fp8.pack(raw, rows_of=_rows)

    def triton(x, w13, w2, s13, s2, w, ids, block_size):
        assert w13 is raw.w13 and w2 is raw.w2 and block_size == (128, 128)
        return reference(x, raw, w, ids).to(x.dtype)

    monkeypatch.setattr(prefill, "experts", triton)
    for answer, passes in ((lambda want: want * 1.01, True), (lambda want: -want, False),
                           (lambda want: want * float("nan"), False), (lambda want: want * 1.2, False)):
        def kernel(x, got_packed, w, ids, answer=answer):
            assert got_packed is packed and w.dtype == torch.float32 and ids.shape[1] == 2
            return answer(reference(x, raw, w, ids).to(x.dtype))

        monkeypatch.setattr(trtllm_fp8, "experts", kernel)
        verdict = trtllm_fp8._probe(raw, packed, top_k=2)
        assert (verdict is None) is passes, verdict


def _kernel_as_documented(x, packed, topk_weights, topk_ids):
    """trtllm_fp8_block_scale_routed_moe's contract, in fp32: gemm1 rows [linear (up) X1; gate X2], out =
    sum_k w_k (silu(X2) * X1) @ gemm2^T, both weights row-shuffled and blocked along K, 128 x 128 block scales."""
    w13, w2 = _as_the_kernel_reads(packed.w13, _rows), _as_the_kernel_reads(packed.w2, _rows)
    inter = w2.shape[2]
    out = torch.zeros(x.shape, dtype=torch.float32)
    for t in range(x.shape[0]):
        for k in range(topk_ids.shape[1]):
            e = topk_ids[t, k]
            x1, x2 = (x[t].float() @ dequantized(w13[e], packed.s13[e]).T).split(inter)
            out[t] += topk_weights[t, k] * ((F.silu(x2) * x1) @ dequantized(w2[e], packed.s2[e]).T)
    return out.to(x.dtype)


def _cfg(kernels="auto", hidden=128):
    cfg = Glm52ModelConfig.reduced_fp8(block=(128, 128))
    cfg.hidden_size = hidden
    cfg.moe_intermediate_size = 128
    cfg.n_routed_experts = 8
    cfg.kernels = kernels
    return cfg


def _block(cfg, raw):
    torch.manual_seed(7)
    block = Glm52SparseMoeBlock(cfg)
    for prm in block.parameters():
        if prm.dtype.is_floating_point:
            prm.data.normal_(0, 0.05)
    block.gate.e_score_correction_bias.data.normal_(0, 0.5)
    exp = block.experts
    for name, t in zip(("gate_up_proj_fp8", "gate_up_proj_scale_inv", "down_proj_fp8", "down_proj_scale_inv"),
                       raw, strict=True):
        getattr(exp, name).data.copy_(t)
    return block


@pytest.fixture
def fused_here(monkeypatch):
    monkeypatch.setattr(moe_mod, "fused_fp8_available", lambda dev, block: True)


def test_a_block_that_took_the_kernel_computes_its_experts(blackwell_stand_in, fused_here, monkeypatch):
    """The repacked experts through the kernel's documented contract give the reference dispatch's output on the
    original ones (the same block without the load hook), at a decode and a prefill token count; the Triton
    paths are never entered."""
    raw = random_experts(8, 128, 128, "cpu", seed=3)
    ref = _block(_cfg(), raw)
    block = _block(_cfg(), raw)
    block.process_weights_after_loading("cpu")
    assert block._trtllm and block._use_fused and blackwell_stand_in == [2]
    assert block.experts.gate_up_proj_fp8.shape == (8, 1, 256, 128)  # the kernel's layout, in place
    seen = []
    monkeypatch.setattr(trtllm_fp8, "experts", lambda *a: seen.append(a[0].shape[0]) or _kernel_as_documented(*a))
    for name in ("_forward_decode", "_dispatch_prefill", "_dispatch_fp8_reference"):
        monkeypatch.setattr(block, name, lambda *a, **k: pytest.fail("a Triton or reference path ran"))
    for tokens in (5, 80):
        x = torch.randn(tokens, 128) * 0.5
        torch.testing.assert_close(block(x), ref(x), rtol=1e-5, atol=1e-5)
    assert seen == [5, 80]


def test_a_block_whose_probe_failed_keeps_triton_and_its_weights(blackwell_stand_in, fused_here, monkeypatch):
    monkeypatch.setattr(trtllm_fp8, "_probe", lambda *a: "relative error 0.9 at 8 tokens, above 0.05")
    raw = random_experts(8, 128, 128, "cpu", seed=3)
    block = _block(_cfg(), raw)
    block.process_weights_after_loading("cpu")
    assert block._use_fused and not block._trtllm
    exp = block.experts
    for name, t in zip(("gate_up_proj_fp8", "gate_up_proj_scale_inv", "down_proj_fp8", "down_proj_scale_inv"),
                       raw, strict=True):
        assert torch.equal(getattr(exp, name), t), name


def test_kernels_reference_never_probes_and_the_hook_is_idempotent(blackwell_stand_in, fused_here):
    raw = random_experts(8, 128, 128, "cpu", seed=3)
    ref = _block(_cfg("reference"), raw)
    ref.process_weights_after_loading("cpu")
    assert not ref._trtllm and blackwell_stand_in == []
    block = _block(_cfg(), raw)
    block.process_weights_after_loading("cpu")
    packed = block.experts.gate_up_proj_fp8.data.clone()
    block.process_weights_after_loading("cpu")  # a second hook must not repack the repacked bytes
    assert block._trtllm and torch.equal(block.experts.gate_up_proj_fp8.data, packed)


def test_a_block_on_the_kernel_skips_the_decode_chain():
    """The decode chain reads [gate; up] in M*'s layout, so a repacked block never takes it."""
    block = _block(_cfg(), random_experts(8, 128, 128, "cpu"))
    block.to(torch.bfloat16)
    restore_fp32_params(block)
    block._use_fused = True
    flat = types.SimpleNamespace(shape=(4, 128), is_cuda=True, dtype=torch.bfloat16)
    assert block._use_decode(flat)
    block._trtllm = True
    assert not block._use_decode(flat)


# ---- sm_100 / sm_103: the kernel itself --------------------------------------------------------------------

cuda_blackwell = pytest.mark.skipif(not BLACKWELL, reason="flashinfer's trtllm fp8 MoE runs on sm_100 / sm_103")
DEV = torch.device("cuda", 0)
# (experts, hidden, per-rank intermediate, top-k): small, mid, GLM-5.3 at TP8
SHAPES = [(8, 512, 128, 2), (64, 2048, 256, 6), (256, 6144, 256, 8)]


def _routing(kind, tokens, e, top_k, hidden, gen):
    if kind == "random":
        return trtllm_fp8.probe_routing(tokens, e, top_k, DEV, gen)
    if kind == "skewed":  # every token on the same experts: one long group per expert
        w, _ = trtllm_fp8.probe_routing(tokens, e, top_k, DEV, gen)
        return w, torch.arange(top_k, device=DEV).repeat(tokens, 1)
    logits = torch.randn(tokens, e, device=DEV, generator=gen)
    return glm_router(logits, torch.randn(e, device=DEV, generator=gen) * 0.1, top_k)


@cuda_blackwell
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "E{}-H{}-I{}-k{}".format(*s))
def test_the_kernel_matches_triton_and_the_dequantized_reference(shape):
    from mstar.utils.fused_moe import prefill

    e, hidden, inter, top_k = shape
    raw = random_experts(e, hidden, inter, DEV)
    packed = trtllm_fp8.pack(raw)
    gen = torch.Generator(device=DEV).manual_seed(0)
    for kind in ("random", "skewed", "glm"):
        for tokens in (1, 7, 64, 300, 2048):
            x = torch.randn(tokens, hidden, device=DEV, generator=gen).to(torch.bfloat16)
            w, ids = _routing(kind, tokens, e, top_k, hidden, gen)
            want = reference(x, raw, w, ids)
            tri = prefill.experts(x, raw.w13, raw.w2, raw.s13, raw.s2, w, ids, block_size=(128, 128))
            got = trtllm_fp8.experts(x, packed, w, ids)
            assert got.shape == x.shape and got.dtype == x.dtype
            case = (kind, tokens, trtllm_fp8.rel_l2(got, tri), trtllm_fp8.rel_l2(got, want),
                    trtllm_fp8.rel_l2(tri, want))
            assert case[2] <= trtllm_fp8.PROBE_TOLERANCE and case[3] <= REFERENCE_TOLERANCE, case


@cuda_blackwell
def test_take_passes_its_probe_on_blackwell():
    raw = random_experts(256, 6144, 256, DEV)
    packed = trtllm_fp8.take(raw, 8, (128, 128))
    assert packed is not None, trtllm_fp8._VERDICTS
    assert packed.w13.shape == (256, 48, 512, 128) and packed.w2.shape == (256, 2, 6144, 128)


@cuda_blackwell
def test_the_kernel_replays_from_a_cuda_graph():
    raw = random_experts(64, 2048, 256, DEV)
    packed = trtllm_fp8.pack(raw)
    gen = torch.Generator(device=DEV).manual_seed(1)
    x = torch.randn(48, 2048, device=DEV, generator=gen).to(torch.bfloat16)
    w, ids = trtllm_fp8.probe_routing(48, 64, 6, DEV, gen)
    eager = trtllm_fp8.experts(x, packed, w, ids)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        trtllm_fp8.experts(x, packed, w, ids)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = trtllm_fp8.experts(x, packed, w, ids)
    x.copy_(torch.randn(48, 2048, device=DEV, generator=gen).to(torch.bfloat16))
    graph.replay()
    torch.testing.assert_close(out, trtllm_fp8.experts(x, packed, w, ids))
    assert not torch.equal(out, eager)


@cuda_blackwell
def test_a_glm_block_on_blackwell_takes_the_kernel_and_matches_the_reference():
    """Against the same block without the load hook, which runs the reference dispatch."""
    raw = random_experts(8, 512, 128, "cpu", seed=3)
    blocks = []
    for hook in (False, True):
        block = _block(_cfg(hidden=512), raw).to(DEV)
        block.to(torch.bfloat16)
        restore_fp32_params(block)
        if hook:
            block.process_weights_after_loading(DEV)
        blocks.append(block)
    ref, block = blocks
    assert block._trtllm and not ref._use_fused, trtllm_fp8._VERDICTS
    for tokens in (3, 100):
        x = torch.randn(tokens, 512, device=DEV).to(torch.bfloat16)
        assert trtllm_fp8.rel_l2(block(x), ref(x)) < REFERENCE_TOLERANCE
