"""Block-fp8 routed experts at a few tokens (fused_moe: runner.experts_fp8, decode.routed_experts_fp8): the
dispatch takes per-pair GEMVs only up to decode.ROUTED_FP8_MAX_TOKENS tokens, on NVIDIA GPUs, for real experts and
shapes the tiles fit, and the grouped W8A8 runner otherwise (CPU). On CUDA: the GEMVs against the dequantized
experts and the grouped runner at Qwen3-Omni's Thinker and Talker shapes (TP 1 and 2), per slot and from a CUDA
graph, and experts_fp8 on both sides of the threshold."""

import pytest
import torch
import torch.nn.functional as F

from mstar.platforms import current_platform

decode = pytest.importorskip("mstar.utils.fused_moe.decode")
runner = pytest.importorskip("mstar.utils.fused_moe.runner")

CUDA = torch.device("cuda", 0)
BLOCK = (128, 128)
# (experts, hidden, per-rank intermediate, top-k)
THINKER, TALKER, THINKER_TP2 = (128, 2048, 768, 8), (128, 1024, 384, 6), (128, 2048, 384, 8)


@pytest.fixture
def nvidia(monkeypatch):
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)


def _w1(e, h, i):
    return (e, 2 * i, h)


@pytest.mark.parametrize("shape", [THINKER, TALKER, THINKER_TP2], ids=["thinker", "talker", "thinker-tp2"])
def test_a_few_tokens_take_the_per_pair_gemvs(nvidia, shape):
    e, h, i, k = shape
    for t in range(1, decode.ROUTED_FP8_MAX_TOKENS + 1):
        assert runner.per_pair_fp8(t, CUDA, _w1(e, h, i), k, BLOCK, False), t
    assert not runner.per_pair_fp8(decode.ROUTED_FP8_MAX_TOKENS + 1, CUDA, _w1(e, h, i), k, BLOCK, False)


def test_padding_slots_rocm_cpu_and_unfit_shapes_keep_the_grouped_runner(nvidia, monkeypatch):
    e, h, i, k = THINKER
    assert not runner.per_pair_fp8(1, CUDA, _w1(e, h, i), k, BLOCK, skip_invalid=True)  # sentinel ids
    assert not runner.per_pair_fp8(1, torch.device("cpu"), _w1(e, h, i), k, BLOCK, False)
    assert not runner.per_pair_fp8(1, CUDA, _w1(e, h, 192), k, BLOCK, False)  # a partial scale block per half
    monkeypatch.setattr(current_platform, "is_cuda", lambda: False)  # ROCm
    assert not runner.per_pair_fp8(1, CUDA, _w1(e, h, i), k, BLOCK, False)


def test_whole_scale_blocks_fit_and_the_tiles_narrow_to_divide():
    for _, h, i, _ in (THINKER, TALKER, THINKER_TP2, (256, 6144, 256, 8)):
        assert decode.routed_fp8_fits(h, i, BLOCK), (h, i)
    assert not decode.routed_fp8_fits(2048, 192, BLOCK) and not decode.routed_fp8_fits(2000, 768, BLOCK)
    assert not decode.routed_fp8_fits(2048, 768, (96, 128))
    assert [decode._fit(256, 384), decode._fit(256, 768), decode._fit(1024, 1024), decode._fit(32, 128)] == [
        128, 256, 1024, 32]


class _Launches:
    """Stands in for a ``@triton.jit`` kernel: records each launch's arguments instead of running it."""

    def __init__(self, fn):
        self.fn, self.calls = fn, []

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.calls.append((args, kwargs))


_TYPES = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32", torch.uint8: "u8",
          torch.int64: "i64", torch.int32: "i32"}


def _compile_sm90(fn, args, kwargs):
    """Compile ``fn`` for sm_90 (no GPU needed) with the types of one recorded launch."""
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    names = fn.arg_names
    values = dict(zip(names, args, strict=False)) | {k: v for k, v in kwargs.items() if k in names}
    consts = {names[i]: values[names[i]] for i in fn.constexprs}
    sig = {n: "constexpr" if n in consts else
           ("*" + _TYPES[v.dtype]) if isinstance(v, torch.Tensor) else "fp32" if isinstance(v, float) else "i64"
           for n, v in values.items()}
    triton.compile(ASTSource(fn=fn, signature=sig, constexprs=consts), target=GPUTarget("cuda", 90, 32))


@pytest.mark.parametrize("shape", [THINKER, TALKER], ids=["thinker", "talker"])
def test_the_per_pair_launches_compile_for_sm90(monkeypatch, shape):
    """The kernels routed_experts_fp8 launches type-check and compile with the arguments it passes (Triton
    compiles a kernel's never-taken branches too, such as the shared expert's)."""
    pytest.importorskip("triton.backends.nvidia")
    e, h, i, k = shape
    gate_up, down = _Launches(decode._gate_up_kernel), _Launches(decode._down_kernel)
    monkeypatch.setattr(decode, "_gate_up_kernel", gate_up)
    monkeypatch.setattr(decode, "_down_kernel", down)
    w13, w2 = torch.zeros(e, 2 * i, h, dtype=torch.uint8), torch.zeros(e, h, i, dtype=torch.uint8)
    s13, s2 = torch.ones(e, 2 * i // 128, h // 128), torch.ones(e, h // 128, i // 128)
    for t in (1, decode.ROUTED_FP8_MAX_TOKENS):
        decode.routed_experts_fp8(torch.zeros(t, h, dtype=torch.bfloat16), w13, s13, w2, s2, torch.ones(t, k),
                                  torch.zeros(t, k, dtype=torch.int64), block_size=BLOCK)
    for launches in (gate_up, down):
        assert launches.calls
        for args, kwargs in launches.calls:
            try:
                _compile_sm90(launches.fn, args, kwargs)
            except (RuntimeError, FileNotFoundError) as exc:  # no ptxas on this host
                if "ptxas" in str(exc).lower():
                    pytest.skip(f"this host can't run ptxas: {exc}")
                raise


# ---- CUDA ----------------------------------------------------------------------------------------------------

cuda = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
                          reason="needs an NVIDIA GPU with e4m3 (the grouped runner's tensor cores)")


def _experts(e, h, i, seed=0):
    """Block-fp8 experts (uint8 bytes, fp32 scales) and their fp32 dequantization."""
    g = torch.Generator(device=CUDA).manual_seed(seed)
    out = []
    for n, kk in ((2 * i, h), (h, i)):
        w = torch.randn(e, n, kk, device=CUDA, generator=g) * 0.02
        tiles = w.reshape(e, n // 128, 128, kk // 128, 128)
        s = (tiles.abs().amax(dim=(2, 4)) / 448.0).clamp_min(1e-12)
        q = (tiles / s[:, :, None, :, None]).to(torch.float8_e4m3fn).reshape(e, n, kk)
        deq = q.float() * s.repeat_interleave(128, 1).repeat_interleave(128, 2)
        out.append((q.view(torch.uint8).contiguous(), s.contiguous(), deq))
    return out


def _reference(x, w13, w2, tw, ids, reduce=True):
    out = torch.zeros(x.shape[0], ids.shape[1], x.shape[1], device=x.device)
    for t in range(x.shape[0]):
        for k in range(ids.shape[1]):
            g, u = (x[t].float() @ w13[ids[t, k]].T).chunk(2)
            out[t, k] = tw[t, k] * ((F.silu(g) * u) @ w2[ids[t, k]].T)
    return out.sum(1) if reduce else out


def _routing(t, e, k, seed):
    g = torch.Generator(device=CUDA).manual_seed(seed)
    ids = torch.rand(t, e, device=CUDA, generator=g).topk(k, dim=-1).indices
    tw = torch.rand(t, k, device=CUDA, generator=g)
    return tw / tw.sum(-1, keepdim=True), ids


def _rel(got, want):
    return ((got.float() - want.float()).norm() / want.float().norm()).item()


@cuda
@pytest.mark.parametrize("shape", [THINKER, TALKER, THINKER_TP2], ids=["thinker", "talker", "thinker-tp2"])
def test_the_per_pair_gemvs_match_the_dequantized_experts_and_the_grouped_runner(shape):
    e, h, i, k = shape
    (w13, s13, d13), (w2, s2, d2) = _experts(e, h, i)
    for t in range(1, decode.ROUTED_FP8_MAX_TOKENS + 2):
        x = torch.randn(t, h, device=CUDA).bfloat16()
        tw, ids = _routing(t, e, k, t)
        got = decode.routed_experts_fp8(x, w13, s13, w2, s2, tw, ids, block_size=BLOCK)
        assert got.shape == x.shape and got.dtype == x.dtype
        want = _reference(x, d13, d2, tw, ids)
        # bf16 activations meet the fp8 weights in fp32: only the output's rounding
        assert _rel(got, want) < 0.01, t
        grouped = runner.fused_experts_fp8(x, w13, w2, s13, s2, tw, ids, block_size=BLOCK)
        assert _rel(got, grouped) < 0.05, t  # the grouped runner quantizes activations to e4m3
        slots = decode.routed_experts_fp8(x, w13, s13, w2, s2, tw, ids, block_size=BLOCK, reduce=False)
        assert slots.shape == (t, k, h) and _rel(slots, _reference(x, d13, d2, tw, ids, reduce=False)) < 0.01


@cuda
def test_experts_fp8_takes_each_path_on_its_side_of_the_threshold(monkeypatch):
    e, h, i, k = THINKER
    (w13, s13, d13), (w2, s2, d2) = _experts(e, h, i, seed=1)
    calls = []
    real = decode.routed_experts_fp8
    monkeypatch.setattr(decode, "routed_experts_fp8", lambda *a, **kw: calls.append(a[0].shape[0]) or real(*a, **kw))
    edge = decode.ROUTED_FP8_MAX_TOKENS
    for t in (1, edge, edge + 1, 40):
        x = torch.randn(t, h, device=CUDA).bfloat16()
        tw, ids = _routing(t, e, k, 100 + t)
        got = runner.experts_fp8(x, w13, w2, s13, s2, tw, ids, block_size=BLOCK)
        assert _rel(got, _reference(x, d13, d2, tw, ids)) < 0.05, t
    assert calls == [1, edge]


@cuda
def test_the_per_pair_gemvs_replay_from_a_cuda_graph():
    e, h, i, k = THINKER
    (w13, s13, _), (w2, s2, _) = _experts(e, h, i, seed=2)
    x = torch.randn(3, h, device=CUDA).bfloat16()
    tw, ids = _routing(3, e, k, 7)
    runner.experts_fp8(x, w13, w2, s13, s2, tw, ids, block_size=BLOCK)  # compile outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = runner.experts_fp8(x, w13, w2, s13, s2, tw, ids, block_size=BLOCK)
    tw2, ids2 = _routing(3, e, k, 8)
    tw.copy_(tw2)
    ids.copy_(ids2)
    graph.replay()
    torch.testing.assert_close(out, runner.experts_fp8(x, w13, w2, s13, s2, tw, ids, block_size=BLOCK))
