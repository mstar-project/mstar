"""Kimi K3.1's decode glue on one kernel each (GPU): the MLA output gate's ``attn * sigmoid(gate)``, and
the NoPE latent's norm, concatenation and cache write through the fused MLA prep with an identity rope
table, against the torch ops they replace."""
import pytest
import torch

from mstar.model.kimi_k3.components.attention import _gate_mul
from mstar.utils import mla_prep

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="the kernels run on the GPU")


def test_the_gate_falls_back_to_torch_off_the_gpu():
    attn, gate = torch.randn(3, 64), torch.randn(3, 64)
    assert torch.equal(_gate_mul(attn, gate), attn * torch.sigmoid(gate))


@gpu
def test_the_gate_kernel_rounds_as_torchs_two_kernels():
    """K3.1's 12 heads x 128 a rank, the gate a strided column block of the packed q_a | kv_a | g GEMM."""
    g = torch.Generator(device="cuda").manual_seed(0)
    for rows in (1, 7, 64):
        packed = torch.randn(rows, 2112 + 1536, device="cuda", generator=g).to(torch.bfloat16)
        gate = packed[:, 2112:]
        attn = torch.randn(rows, 1536, device="cuda", generator=g).to(torch.bfloat16)
        want = attn * torch.sigmoid(gate)
        got = _gate_mul(attn, gate)
        mismatch = (got != want).float().mean().item()
        # tl.sigmoid and torch's may differ in fp32's last bit, which a bf16 rounding rarely shows
        assert mismatch < 1e-3 and (got.float() - want.float()).abs().max().item() <= 2 ** -6, mismatch


@gpu
def test_the_identity_rope_prep_writes_the_nope_latent_and_passes_q_pe_through():
    """``[rmsnorm(kv_a) | k_pe]`` into the cache slots and ``q_pe`` unchanged: the decoupled dims keep
    their bits (cos 1, sin 0), the norm is the RMSNorm's within a bf16 rounding."""
    from mstar.model.components.norm import RMSNorm

    torch.manual_seed(0)
    T, H, Q, L, R, nope = 5, 12, 1536, 512, 64, 128
    fused = torch.randn(T, Q + L + R + 1536, device="cuda").to(torch.bfloat16)[:, : Q + L + R]
    q = torch.randn(T, H, nope + R, device="cuda").to(torch.bfloat16)
    norm = RMSNorm(L, eps=1e-6).cuda().to(torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_((1 + 0.1 * torch.randn(L, device="cuda")).to(torch.bfloat16))
    cache = torch.zeros(4, 16, L + R, device="cuda", dtype=torch.bfloat16)
    pages = torch.tensor([3, 0, 0, 2, 1], device="cuda")
    offsets = torch.tensor([5, 0, 9, 15, 2], device="cuda")
    table = torch.cat([torch.ones(1, R // 2), torch.zeros(1, R // 2)], dim=1).cuda()
    positions = torch.zeros(T, dtype=torch.int32, device="cuda")
    q_pe = mla_prep.kv_write_q_rope(fused, norm.weight, norm.variance_epsilon, table, positions, q, cache,
                                    pages, offsets)
    assert torch.equal(q_pe, q[..., nope:])
    rows = cache[pages, offsets]
    assert torch.equal(rows[:, L:], fused[:, Q + L:])
    want = norm(fused[:, Q:Q + L].contiguous())
    assert (rows[:, :L].float() - want.float()).abs().max().item() <= 2 ** -6 * want.float().abs().max().item()


def _rcp448() -> torch.Tensor:
    """``rcp.approx.ftz(448)`` as this GPU computes it, the factor TRT-LLM's MXFP8 scale takes."""
    import triton
    import triton.language as tl

    from mstar.utils.fused_moe.mxfp8_cast import _rcp_approx_ftz

    @triton.jit
    def k(out):
        tl.store(out, _rcp_approx_ftz(tl.full([1], 448.0, tl.float32)).reshape([]))

    out = torch.empty(1, device="cuda")
    k[(1,)](out)
    return out


def _mxfp8_reference(z: torch.Tensor, rcp448: torch.Tensor):
    """TRT-LLM's cvt_warp_fp16_to_mxfp8 in torch: amax * rcp(448) rounded up to a power of two (2^-127 at or
    below it, whose denormal float TRT-LLM's ftz test reads as 0: the block's values times 0), values times its
    ftz reciprocal, clamped, rounded to e4m3."""
    v = z.float().reshape(z.shape[0], -1, 32)
    bits = (v.abs().amax(-1) * rcp448).view(torch.int32)
    exp, mant = (bits >> 23) & 0xFF, bits & 0x7FFFFF
    e = torch.where(exp == 0, (mant > 0x400000).int(), exp + (mant != 0).int()).clamp(max=254)
    scale = torch.where(e == 0, torch.zeros_like(v[..., 0]), ((254 - e) << 23).view(torch.float32))
    y = v * scale[..., None]
    y = y.clamp(-448.0, 448.0)
    return y.to(torch.float8_e4m3fn).reshape(z.shape), e.to(torch.uint8)


def _packed_rows(rows: int, e: int, latent: int, rest: int, g) -> torch.Tensor:
    """fp32 ``[logits | z | gate_up]`` rows with z's groups at every scale: zeros, tiny, large, a power of two."""
    h = torch.randn(rows, e + latent + rest, device="cuda", generator=g) * 3
    z = h[:, e:e + latent]
    z[0, :32] = 0
    z[0, 32:64] *= 1e-37
    z[-1, 64:96] *= 1e4
    z[-1, 96:128] = 448.0 * 2.0 ** -3
    return h


@gpu
def test_the_split_cast_rounds_gate_up_and_quantizes_z_as_trtllm_does():
    """K3.1's packed MoE-input rows (logits | z | gate_up): gate_up's bf16 bits, z's MXFP8 values and scales as
    TRT-LLM's algorithm gives them from z's bf16 rounding, and within e4m3's rounding of z."""
    from mstar.utils.fused_moe import mxfp8_cast

    g = torch.Generator(device="cuda").manual_seed(0)
    rcp = _rcp448()
    for rows, e, latent, rest in ((1, 112, 1024, 1536), (5, 112, 1024, 1536), (64, 96, 512, 768)):
        h = _packed_rows(rows, e, latent, rest, g)
        q, sf, gate_up = torch.ops.mstar.split_cast_mxfp8(h, e, latent)
        assert gate_up.is_contiguous() and torch.equal(gate_up, h[:, e + latent:].to(torch.bfloat16))
        z = h[:, e:e + latent].to(torch.bfloat16)
        want_q, want_sf = _mxfp8_reference(z, rcp)
        assert torch.equal(sf, want_sf), (sf != want_sf).nonzero()[:4]
        got_q, want_q = q.view(torch.uint8), want_q.view(torch.uint8)
        assert torch.equal(got_q, want_q), (got_q != want_q).nonzero()[:4]
        # e4m3 keeps 3 mantissa bits (half a step: 2^-4 relative) down to 2^-6, then steps of 2^-9: in units of
        # the group's scale, half of either
        scale = torch.exp2(sf.float() - 127)[..., None]
        back = q.float().reshape(rows, -1, 32) * scale
        want = z.float().reshape(rows, -1, 32)
        live = (sf > 0)[..., None].expand_as(back)
        bound = 2.0 ** -4 * want.abs() + 2.0 ** -10 * scale
        assert bool(((back - want).abs() <= bound)[live].all())
        assert mxfp8_cast.supported(h, latent)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
                    reason="FlashInfer's MXFP8 quantization runs on SM100 and newer")
def test_the_split_cast_matches_flashinfers_mxfp8_quantize_bit_for_bit():
    """On a Blackwell GPU, against ``mxfp8_quantize(z, is_sf_swizzled_layout=False)`` itself, the call it replaces."""
    from flashinfer.quantization.fp8_quantization import mxfp8_quantize

    g = torch.Generator(device="cuda").manual_seed(1)
    for rows in (1, 3, 64):
        h = _packed_rows(rows, 112, 1024, 1536, g)
        q, sf, _ = torch.ops.mstar.split_cast_mxfp8(h, 112, 1024)
        want_q, want_sf = mxfp8_quantize(h[:, 112:1136].to(torch.bfloat16), is_sf_swizzled_layout=False)
        assert torch.equal(sf.reshape(-1), want_sf.reshape(-1).view(torch.uint8))
        assert torch.equal(q.view(torch.uint8), want_q.view(torch.uint8).reshape(q.shape))
