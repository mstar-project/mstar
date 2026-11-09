"""The vendored Marlin MoE (``mstar.utils.marlin_moe``) for the 4-bit routed experts: its weight layout is vLLM's
bit for bit per expert (digests of vLLM 0.33.0's reference functions on fixed inputs) and reads back exactly; the
block size follows vLLM's rule; a block's conversion keeps the Triton kernel when the numerics probe disagrees and
frees the checkpoint copy when it agrees; on a GPU (the op built there) every token count, TP-style and with the
expert-parallel sentinel, matches the experts dequantized in fp32, captured in a CUDA graph."""

import hashlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from mstar.utils import marlin_moe as M

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _digest(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


def test_each_expert_is_vllms_dense_marlin_layout_bit_for_bit():
    """The fixed inputs the dense Marlin test pins (digests computed with vLLM 0.33.0's marlin_weights,
    marlin_permute_scales, marlin_zero_points and mxfp4_marlin_process_scales), as expert 1 of a stack of two."""
    k, n, g = 256, 128, 64
    kk, nn_, gg = torch.arange(k)[:, None], torch.arange(n)[None, :], torch.arange(k // g)[:, None]
    codes = ((kk * 31 + nn_ * 7 + 3) % 16).to(torch.uint8)                         # (K, N)
    s = ((gg * 37 + nn_) % 200 + 1).to(torch.bfloat16)                             # (K/g, N)
    z = ((gg * 5 + nn_ * 3) % 16).to(torch.uint8)
    e8 = ((torch.arange(n)[:, None] * 3 + torch.arange(k // 32)[None, :] * 11) % 40 + 100).to(torch.uint8)
    pair = lambda t: torch.stack([torch.zeros_like(t), t])  # noqa: E731
    assert _digest(M.repack_experts(pair(codes))[1]) == "4b0d84388ea59149"
    assert _digest(M.permute_scales(pair(s))[1]) == "6341d8d9255a6f11"
    assert _digest(M.zero_points(pair(z))[1]) == "a4d1b990568ff00b"
    packed = (codes.t()[:, 0::2] | (codes.t()[:, 1::2] << 4)).contiguous()       # (N, K/2), as stored
    held = M.prepare_experts(M.KIND_E2M1, pair(packed), pair(e8))
    assert _digest(held["weight"][1]) == "4b0d84388ea59149" and _digest(held["scale"][1]) == "8460cded5b686f89"


def _stack(kind, group, e=3, n=192, k=512, seed=0):
    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 16, (e, n, k), dtype=torch.uint8, generator=g)
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous()
    if kind == M.KIND_E2M1:
        scale = torch.randint(118, 126, (e, n, k // group), dtype=torch.uint8, generator=g)
    else:
        scale = (torch.rand(e, n, k // group, generator=g) * 0.02 + 0.005).to(torch.bfloat16)
    zu = torch.randint(0, 16, (e, n, k // group), dtype=torch.uint8, generator=g) if kind == M.KIND_ZP else None
    zero = None
    if zu is not None:
        rows = zu.view(e, n // 8, 8, k // group).to(torch.int64)
        zero = sum(rows[:, :, j] << (4 * j) for j in range(8)).to(torch.int32)    # 8 rows a word, as stored
    return codes, packed, scale, zu, zero


@pytest.mark.parametrize("kind, group", [(M.KIND_SYM, 128), (M.KIND_ZP, 32), (M.KIND_ZP, 64), (M.KIND_E2M1, 32)])
def test_marlin_copy_reads_back_the_checkpoint_weights(kind, group):
    codes, packed, scale, zu, zero = _stack(kind, group)
    held = M.prepare_experts(kind, packed, scale, zero, chunk=2)
    got = M.dequantize_experts(kind, group, held["weight"], held["scale"], held.get("zero"))
    if kind == M.KIND_E2M1:
        table = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
        want = table[codes.long()].view(*codes.shape[:2], -1, 32) * torch.exp2(scale.float() - 127.0)[..., None]
    else:
        z = zu.float()[..., None] if zu is not None else 8.0
        want = (codes.float().view(*codes.shape[:2], -1, group) - z) * scale.float()[..., None]
    assert torch.equal(got, want.reshape(codes.shape))
    # chunking doesn't change the layout
    whole = M.prepare_experts(kind, packed, scale, zero, chunk=64)
    assert all(torch.equal(whole[key], held[key]) for key in held)


def test_block_size_follows_vllms_rule():
    # Kimi K3.1 at TP8 (896 experts, top-16): decode on 8-row blocks, 512 tokens (9 a expert) on 16, prefill on 64
    assert M.block_size(16, 896, 16) == 8 and M.block_size(512, 896, 16) == 16
    assert M.block_size(8192, 896, 16) == 64
    # EP8: 112 local experts of 896; the token count scales to this rank's share (the same per-expert load)
    assert M.block_size(16, 112, 16, global_experts=896) == 8
    assert M.block_size(512, 112, 16, global_experts=896) == 16
    assert M.block_size(8192, 112, 16, global_experts=896) == 64


def _block(kind, group, activation="silu"):
    """A stand-in QuantExperts block holding a format's stacked experts, as the loader leaves them."""
    from mstar.model.components.int4_formats import MXFP4, W4A16Format

    e, inter, hidden = 4, 128, 256
    c1, p1, s1, _, z1 = _stack(kind, group, e, 2 * inter, hidden, seed=1)
    c2, p2, s2, _, z2 = _stack(kind, group, e, hidden, inter, seed=2)
    experts = nn.Module()
    views = {}
    for proj, p, s, z in (("gate_up_proj", p1, s1, z1), ("down_proj", p2, s2, z2)):
        setattr(experts, f"{proj}_packed", nn.Parameter(p, requires_grad=False))
        setattr(experts, f"{proj}_scale", nn.Parameter(s, requires_grad=False))
        views[f"{proj}_packed"] = (torch.uint8, None, proj, "weight_packed")
        views[f"{proj}_scale"] = (torch.uint8 if kind == M.KIND_E2M1 else torch.bfloat16, None, proj, "weight_scale")
        if z is not None:
            setattr(experts, f"{proj}_zero_point", nn.Parameter(z, requires_grad=False))
            views[f"{proj}_zero_point"] = (torch.int32, None, proj, "weight_zero_point")
    fmt = MXFP4 if kind == M.KIND_E2M1 else W4A16Format(group=group, symmetric=kind == M.KIND_SYM)

    def buffers(proj, index):
        return {suffix: getattr(experts, name)[index] for name, (_, _, p, suffix) in views.items() if p == proj}

    return SimpleNamespace(experts=experts, experts_packed=fmt, _experts_views=views, _expert_buffers=buffers,
                           _experts_dims=lambda: (e, 2 * inter, hidden), _experts_activation=activation,
                           num_experts=e, num_experts_per_tok=2)


@pytest.mark.parametrize("agree", [True, False])
def test_conversion_probes_the_triton_kernel_and_frees_the_checkpoint_copy(agree, monkeypatch):
    """Off CUDA, Marlin's CPU reference stands in for the op and a reference for the Triton kernel: agreeing, the
    block gets Marlin and its checkpoint copy is freed (one copy); disagreeing, it keeps both and the Triton
    kernel."""
    import mstar.utils.fused_moe.w4a16 as triton_moe

    block = _block(M.KIND_ZP, 32)
    held = {name: getattr(block.experts, name).data.clone() for name in block._experts_views}

    def triton_stub(x, w1, w2, s1, s2, z1, z2, weights, ids, kind, group, activation, skip):
        ref = M.fused_experts_marlin(x, *(converted["w1"], converted["w2"], converted["s1"], converted["s2"]),
                                     converted["z1"], converted["z2"], weights, ids, kind, group, activation, skip, 0)
        return ref if agree else ref + 1.0

    converted = {}
    real_prepare = M.prepare_experts

    def spy_prepare(kind, packed, scale, zero=None, **kw):
        out = real_prepare(kind, packed, scale, zero, **kw)
        tag = "1" if "w1" not in converted else "2"
        converted.update({f"w{tag}": out["weight"], f"s{tag}": out["scale"], f"z{tag}": out.get("zero")})
        return out

    monkeypatch.setattr(M, "available", lambda kind, group, device: "stub")
    monkeypatch.setattr(M, "prepare_experts", spy_prepare)
    monkeypatch.setattr(triton_moe, "fused_experts_w4a16", triton_stub)
    kernel = M.MarlinExperts.convert(block, torch.device("cpu"))
    if agree:
        assert kernel is not None
        assert all(getattr(block.experts, name).numel() == 0 for name in block._experts_views)
    else:
        assert kernel is None
        assert all(torch.equal(getattr(block.experts, name).data, held[name]) for name in block._experts_views)


# ---------------------------------------------------------------- CUDA: the vendored op against the reference
@cuda
@pytest.mark.parametrize("kind, group", [(M.KIND_E2M1, 32), (M.KIND_SYM, 128), (M.KIND_ZP, 32)])
@pytest.mark.parametrize("skip", [False, True])
def test_cuda_marlin_moe_matches_the_dequantized_experts(kind, group, skip):
    from mstar.utils.situ import SiTU

    dev = torch.device("cuda")
    assert M.available(kind, group, dev), "the vendored Marlin MoE did not build here (see the log)"
    e, inter, hidden, top_k = 16, 384, 1024, 4
    _, p1, s1, _, z1 = _stack(kind, group, e, 2 * inter, hidden, seed=3)
    _, p2, s2, _, z2 = _stack(kind, group, e, hidden, inter, seed=4)
    h1 = M.prepare_experts(kind, p1.to(dev), s1.to(dev), None if z1 is None else z1.to(dev))
    h2 = M.prepare_experts(kind, p2.to(dev), s2.to(dev), None if z2 is None else z2.to(dev))
    situ = SiTU(4.0, 25.0)
    args = (h1["weight"], h2["weight"], h1["scale"], h2["scale"], h1.get("zero"), h2.get("zero"))
    for m in (1, 8, 64, 300, 1024):
        g = torch.Generator(device=dev).manual_seed(m)
        x = (torch.randn(m, hidden, generator=g, device=dev)).to(torch.bfloat16)
        ids = torch.rand(m, e + (4 if skip else 0), generator=g, device=dev).topk(top_k, -1).indices
        ids = torch.where(ids < e, ids, torch.full_like(ids, e)).to(torch.int32)   # >= e: another rank's
        wts = torch.rand(m, top_k, generator=g, device=dev).softmax(-1)
        got = M.fused_experts_marlin(x, *args, wts, ids, kind, group, situ.spec, skip, 0)
        ref = M.fused_experts_marlin(x.cpu(), *(None if t is None else t.cpu() for t in args), wts.cpu(), ids.cpu(),
                                     kind, group, situ.spec, skip, 0)
        rel = ((got.float().cpu() - ref.float()).norm() / ref.float().norm().clamp_min(1e-9)).item()
        assert rel < 1e-2, (m, rel)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = M.fused_experts_marlin(x, *args, wts, ids, kind, group, situ.spec, skip, 0)
        graph.replay()
        torch.cuda.synchronize()
        # Not bit-exact once an expert spans several blocks (job 1597: 300+ tokens here): moe_align orders an
        # expert's tokens by atomics, and Marlin splits a tile's K reduction by the tile's place in its work list,
        # so a token's sum can split differently run to run. Within a GEMM's rounding of each other.
        if m <= 64:
            assert torch.equal(out, got), m
        else:
            assert ((out.float() - got.float()).norm() / got.float().norm()).item() < 2e-3, m


@cuda
@pytest.mark.parametrize("kind, group", [(M.KIND_E2M1, 32), (M.KIND_ZP, 32)])
def test_cuda_a_block_converts_after_its_probe_agrees(kind, group):
    """The load-time conversion on a GPU: the Triton kernel and Marlin MoE agree on the probe, the block keeps one
    copy, and its experts (SiTU) match the checkpoint's weights dequantized in fp32."""
    from mstar.utils.situ import SiTU

    situ = SiTU(4.0, 25.0)
    block = _block(kind, group, activation=situ)
    dev = torch.device("cuda")
    assert M.available(kind, group, dev)
    block.experts.to(dev)
    source = {proj: {k: v.clone() for k, v in block._expert_buffers(proj, slice(None)).items()}
              for proj in ("gate_up_proj", "down_proj")}
    kernel = M.MarlinExperts.convert(block, dev)
    assert kernel is not None and all(getattr(block.experts, n).numel() == 0 for n in block._experts_views)
    e, two_inter, hidden = block._experts_dims()
    x = torch.randn(40, hidden, device=dev).to(torch.bfloat16)
    ids = torch.rand(40, e, device=dev).topk(2, -1).indices.to(torch.int32)
    wts = torch.rand(40, 2, device=dev).softmax(-1)
    got = kernel(x, wts, ids, False)
    # the reference from the checkpoint copy taken before the conversion freed it
    held = {proj: M.prepare_experts(kind, b["weight_packed"].cpu(), b["weight_scale"].cpu(),
                                    None if b.get("weight_zero_point") is None else b["weight_zero_point"].cpu())
            for proj, b in source.items()}
    g, d = held["gate_up_proj"], held["down_proj"]
    ref = M.fused_experts_marlin(x.cpu(), g["weight"], d["weight"], g["scale"], d["scale"], g.get("zero"),
                                 d.get("zero"), wts.cpu(), ids.cpu(), kind, group, situ.spec, False, 0)
    assert ((got.float().cpu() - ref.float()).norm() / ref.float().norm()).item() < 1e-2
