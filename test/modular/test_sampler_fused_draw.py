"""A kernel that draws its own tokens in the forward takes the graph sampler's per-row parameters
(``fused_draw_args``) and moves the offsets past its draws (``advance_offsets``). Its tokens and the
offsets commit persists have to equal what that many ``sample`` calls would give: otherwise a row's
next step draws from random numbers it already used."""
import pytest
import torch

from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.engine.resources.sampler.utils import SamplerBuffers, SamplingConfig, fused_sample

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (Triton)")
requires_hopper = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="needs a Hopper GPU (Gluon)",
)

_CONFIGS = {
    "a": SamplingConfig(temperature=0.9, top_k=50, top_p=1.0),
    "b": SamplingConfig(temperature=0.0),
    "c": SamplingConfig(temperature=1.2, top_k=0, top_p=0.8),
}
for _seed, _config in enumerate(_CONFIGS.values(), start=7):
    _config.set_seed(_seed)


def _sampler(device: str):
    """A graph sampler over rows [a, b, c] padded to 4, its offsets moved off zero."""
    bufs = SamplerBuffers.allocate(4, torch.device(device), cg_slots=1)
    for rid, config in _CONFIGS.items():
        bufs.register_request(rid, config)
    rids = list(_CONFIGS)
    bufs.gather_static(rids, 4, 0)
    bufs.gather_dynamic(rids, 4, 0)
    sampler = bufs.sampler_for(4, 0)
    sampler.offset_buf.add_(torch.tensor([3, 5, 11, 0], device=device))
    return bufs, sampler, rids


def test_an_eager_step_has_no_fused_draw():
    resource = SamplerResource(vocab_size=2048, enable_repetion_penalty=False, device=torch.device("cpu"))
    assert resource.fused_draw_args() is None, "an eager step draws through the eager sampler, not fused_sample"


class _StubGraphSampler:
    """Hands back a marker for any fused draw."""

    def fused_draw_args(self, apply_filters):
        return "draw"


def _leased(**flags):
    """A resource on a leased step whose graph sampler can draw, with ``flags`` set on it."""
    resource = SamplerResource(vocab_size=2048, enable_repetion_penalty=False, device=torch.device("cpu"))
    resource._cg_sampler, resource._apply_filters_this_step = _StubGraphSampler(), True
    resource._apply_penalty_this_step = resource._keep_last_token = False
    for name, value in flags.items():
        setattr(resource, name, value)
    return resource


def test_a_leased_step_hands_over_its_draw():
    assert _leased().fused_draw_args() == "draw", "a leased step with no penalty and no kept token lost its fused draw"


def test_a_step_with_the_penalty_has_no_fused_draw():
    assert _leased(_apply_penalty_this_step=True).fused_draw_args() is None, "the kernel's draw has no penalty"


def test_a_step_that_keeps_its_last_token_has_no_fused_draw():
    assert _leased(_keep_last_token=True).fused_draw_args() is None, "nothing copies its tokens to last_token_buf"


def test_a_cpu_graph_sampler_has_no_fused_draw():
    _, sampler, _ = _sampler("cpu")
    assert sampler.fused_draw_args(apply_filters=True) is None, "fused_sample runs only on CUDA"


@requires_cuda
def test_fused_draws_then_one_advance_equal_as_many_sample_calls():
    logits = torch.randn(15, 4, 2048, device="cuda")
    bufs, sampler, rids = _sampler("cuda")
    padded = [*rids, "pad"]
    want = torch.stack([sampler.sample(padded, logits[g]) for g in range(15)])
    want_offsets = sampler.offset_buf.clone()

    bufs, sampler, rids = _sampler("cuda")
    args = sampler.fused_draw_args(apply_filters=True)
    got = torch.stack([fused_sample(logits[g], args.temperature, args.top_k, args.top_p, args.seed, args.offset + g)
                       for g in range(15)])
    sampler.advance_offsets(15)
    assert torch.equal(got, want), "the fused draws picked other tokens than the sample calls"
    assert torch.equal(sampler.offset_buf, want_offsets), "the next step would reuse this step's random numbers"


@requires_cuda
def test_a_step_without_filters_draws_with_them_off():
    logits = torch.randn(4, 2048, device="cuda")
    _, sampler, rids = _sampler("cuda")
    want = sampler.sample([*rids, "pad"], logits, apply_filters=False)
    _, sampler, _ = _sampler("cuda")
    args = sampler.fused_draw_args(apply_filters=False)
    got = fused_sample(logits, args.temperature, args.top_k, args.top_p, args.seed, args.offset)
    assert torch.equal(got, want), "top-k and top-p have to be off when the step turns the filters off"


@requires_hopper
def test_draw_row_picks_the_tokens_fused_sample_picks():
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl

    from mstar.engine.resources.sampler.fused_draw import draw_row, layout

    @gluon.jit
    def draw(L, T, K, P, S, O, out, V: gl.constexpr, LAY: gl.constexpr):
        r = gl.program_id(0)
        gl.store(out + r, draw_row(L + r * V, gl.load(T + r), gl.load(K + r), gl.load(P + r), gl.load(S + r),
                                     gl.load(O + r), V, V, LAY))

    torch.manual_seed(1)
    rows = 64
    logits = torch.randn(rows, 2048, device="cuda").bfloat16()
    logits[::4, :64] = logits[::4, :1]  # ties at the top
    temperature = torch.rand(rows, device="cuda") * 1.5
    temperature[::5] = 0
    top_k = torch.randint(0, 2100, (rows,), device="cuda", dtype=torch.int32)
    top_k[::3] = 0
    top_p = torch.rand(rows, device="cuda") * 0.6 + 0.4
    top_p[1::3] = 1.0
    seed = torch.randint(0, 2**40, (rows,), device="cuda")
    for offset in range(16):
        off = torch.full((rows,), offset, device="cuda", dtype=torch.long)
        got = torch.empty(rows, device="cuda", dtype=torch.long)
        draw[(rows,)](logits, temperature, top_k, top_p, seed, off, got, 2048, layout(), num_warps=4)
        want = fused_sample(logits, temperature, top_k, top_p, seed, off)
        assert torch.equal(got, want), f"draw_row picked other tokens than fused_sample at offset {offset}"
