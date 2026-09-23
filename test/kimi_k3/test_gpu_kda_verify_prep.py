"""GPU: the fused verify preparation (``kda_verify_prep``, one launch) is bit-identical to the torch
glue it replaced: windows and prefixes gathered by slot, fp32 tap-order convolutions with SiLU, the no-op
masking past the accepted length (clamped, as padding rows read the sink), the pool's window
rewritten after the prefix and the block saved as the next prefix with its raw gates and betas."""
import pytest
import torch
import torch.nn.functional as F

from mstar.engine.resources.linear_attn.kda_kernels import SpecBlocks
from mstar.engine.resources.linear_attn.kda_spec_prep import kda_verify_prep

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
DEV = torch.device("cuda")
SLOTS, W = 6, 4


def reference(qkv, g_raw, beta_raw, conv_state, spec, slots, conv_w, rows, k1, h, d):
    """The previous implementation's torch glue, on copies of the pool blocks; the prefix part is
    padded to the pool's ``kp`` slots, the block has ``k1`` tokens."""
    conv_state, prefix, g_blk, b_blk = (x.clone() for x in (conv_state, spec.prefix, spec.g, spec.beta))
    conv_w = conv_w.float()
    w = conv_w.shape[1]
    kp = prefix.shape[1]
    slots = slots.to(torch.long)
    win = conv_state.index_select(0, slots).transpose(1, 2).float()
    pre = prefix.index_select(0, slots).float()
    plen = spec.length.index_select(0, slots)[:, 0].clamp(0, kp)
    blk = qkv.view(rows, k1, -1).float()

    def conv(x):
        t = x.shape[1] - (w - 1)
        y = x[:, 0:t] * conv_w[:, 0]
        for j in range(1, w):
            y = y + x[:, j:j + t] * conv_w[:, j]
        return F.silu(y)

    combined = torch.cat([win, pre], dim=1)
    idx = plen.to(torch.long)[:, None] + torch.arange(w - 1, device=DEV)[None, :]
    win_after = torch.gather(combined, 1, idx[:, :, None].expand(-1, -1, combined.shape[-1]))
    y_pre = conv(combined).to(qkv.dtype)
    y_blk = conv(torch.cat([win_after, blk], dim=1)).to(qkv.dtype)
    conv_state.index_copy_(0, slots, win_after.transpose(1, 2).to(conv_state.dtype))
    p = h * d
    pad = torch.arange(kp, device=DEV)[None, :] >= plen[:, None]  # [rows, kp]
    y_pre = torch.where(pad[:, :, None], torch.zeros_like(y_pre), y_pre)  # q too (its value there is irrelevant)
    y = torch.cat([y_pre, y_blk], dim=1)  # [rows, kp + k1, 3P]
    g_pre = torch.where(pad[:, :, None, None], torch.full_like(g_blk[:1, :1], -1e4).expand(rows, kp, h, d),
                        g_blk.index_select(0, slots))
    b_pre = torch.where(pad[:, :, None], torch.full_like(b_blk[:1, :1], -1e4).expand(rows, kp, h),
                        b_blk.index_select(0, slots))
    t2 = kp + k1
    g = torch.cat([g_pre.to(qkv.dtype), g_raw.reshape(rows, k1, h, d)], dim=1).reshape(rows * t2, p)
    beta = torch.cat([b_pre.to(qkv.dtype), beta_raw.view(rows, k1, h)], dim=1).reshape(rows * t2, h)
    # the block fills the leading slots of the next prefix; the slots beyond it keep what they held
    new_prefix, new_g, new_b = (x.index_select(0, slots).clone() for x in (prefix, g_blk, b_blk))
    new_prefix[:, :k1] = blk.to(prefix.dtype)
    new_g[:, :k1] = g_raw.reshape(rows, k1, h, d).to(g_blk.dtype)
    new_b[:, :k1] = beta_raw.view(rows, k1, h).to(b_blk.dtype)
    prefix.index_copy_(0, slots, new_prefix)
    g_blk.index_copy_(0, slots, new_g)
    b_blk.index_copy_(0, slots, new_b)
    y = y.reshape(rows * t2, 3, p)
    return (y[:, 0], y[:, 1], y[:, 2], g, beta, (plen - 1).to(torch.int32)), (conv_state, prefix, g_blk, b_blk), pad


@pytest.mark.parametrize("h,d,k1,kp,rows,strided", [
    (4, 32, 8, 8, 5, False), (2, 48, 6, 6, 3, True), (4, 64, 4, 4, 1, False),
    (2, 32, 2, 8, 3, True), (2, 32, 1, 8, 2, False)])
def test_prep_matches_the_torch_glue(h, d, k1, kp, rows, strided):
    """``kp`` prefix slots (the pool's largest block + 1) and a block of ``k1`` tokens, shorter when the
    block length follows the batch size, down to a single token; ``strided`` hands the gates and betas
    over as column slices of a wider matrix, as the layer's merged projection does."""
    gen = torch.Generator(device=DEV).manual_seed(0)
    p = h * d
    rnd = lambda *shape, s=1.0: (torch.randn(*shape, device=DEV, generator=gen) * s).to(torch.bfloat16)  # noqa: E731
    qkv, g_raw, beta_raw = rnd(rows * k1, 3 * p), rnd(rows * k1, h, d, s=2.0), rnd(rows * k1, h, s=2.0)
    if strided:
        wide = rnd(rows * k1, 5 + p + 3 + h + 2, s=2.0)
        wide[:, 5:5 + p] = g_raw.reshape(rows * k1, p)
        wide[:, 5 + p + 3:5 + p + 3 + h] = beta_raw
        g_raw, beta_raw = wide[:, 5:5 + p].view(rows * k1, h, d), wide[:, 5 + p + 3:5 + p + 3 + h]
        assert not g_raw.is_contiguous() and not beta_raw.is_contiguous()
    conv_state, prefix = rnd(SLOTS, 3 * p, W - 1), rnd(SLOTS, kp, 3 * p)
    spec_g, spec_beta = rnd(SLOTS, kp, h, d, s=2.0), rnd(SLOTS, kp, h, s=2.0)
    # accepted lengths: none, some, all, and the sink's garbage above and below the range
    length = torch.tensor([[0], [3], [kp], [kp + 5], [-2], [1]], dtype=torch.int32, device=DEV)
    slots = torch.tensor([4, 0, 5, 2, 1][:rows], dtype=torch.int32, device=DEV)
    conv_w = rnd(3 * p, W, s=0.3)
    spec = SpecBlocks(prefix, spec_g, spec_beta, length)
    want, want_pool, pad = reference(qkv, g_raw, beta_raw, conv_state, spec, slots, conv_w, rows, k1, h, d)
    got = kda_verify_prep(qkv, g_raw, beta_raw, conv_state, spec, slots, conv_w, rows, k1, h, d)
    torch.cuda.synchronize()
    for name, a, b in zip(("q", "k", "v"), got[:3], want[:3], strict=True):
        assert a.shape == b.shape == (rows * (kp + k1), p), (name, a.shape)
        assert torch.equal(a, b), (name, (a.float() - b.float()).abs().max(), (a != b).float().mean())
    assert torch.equal(got[3], want[3]), "raw gates"
    assert torch.equal(got[4], want[4]), "raw betas"
    assert got[5].tolist() == want[5].tolist() == [max(0, min(int(length[s, 0]), kp)) - 1 for s in slots.tolist()]
    names = ("conv window", "prefix", "prefix gates", "prefix betas")
    for name, a, b in zip(names, (conv_state, prefix, spec_g, spec_beta), want_pool, strict=True):
        assert torch.equal(a, b), name
    # the no-op tokens: k = v = 0 and -1e4 raw gate / beta exactly where the prefix is past its length
    padded = torch.cat([pad, torch.zeros(rows, k1, dtype=torch.bool, device=DEV)], dim=1).reshape(-1)
    assert torch.all(got[1][padded] == 0) and torch.all(got[2][padded] == 0)
    assert torch.all(got[3][padded] == torch.tensor(-1e4, dtype=torch.bfloat16, device=DEV))
    assert torch.all(got[4][padded] == torch.tensor(-1e4, dtype=torch.bfloat16, device=DEV))


def test_prep_is_capturable():
    """Captured and replayed, the launch produces what the eager one did and leaves the pool the same."""
    h, d, k1, rows = 2, 32, 8, 3
    p = h * d
    gen = torch.Generator(device=DEV).manual_seed(1)
    rnd = lambda *shape: torch.randn(*shape, device=DEV, generator=gen).to(torch.bfloat16)  # noqa: E731
    qkv, g_raw, beta_raw = rnd(rows * k1, 3 * p), rnd(rows * k1, p), rnd(rows * k1, h)
    spec = SpecBlocks(rnd(SLOTS, k1, 3 * p), rnd(SLOTS, k1, h, d), rnd(SLOTS, k1, h),
                      torch.full((SLOTS, 1), 2, dtype=torch.int32, device=DEV))
    conv_state, conv_w = rnd(SLOTS, 3 * p, W - 1), rnd(3 * p, W)
    slots = torch.tensor([1, 3, 5], dtype=torch.int32, device=DEV)
    args = (qkv, g_raw, beta_raw, conv_state, spec, slots, conv_w, rows, k1, h, d)
    pool = (conv_state, spec.prefix, spec.g, spec.beta)  # everything the launch rewrites
    before = [x.clone() for x in pool]

    def restore():
        torch.cuda.synchronize()
        for x, y in zip(pool, before, strict=True):
            x.copy_(y)
        torch.cuda.synchronize()

    want = [x.clone() for x in kda_verify_prep(*args)]
    pool_after_eager = [x.clone() for x in pool]
    restore()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        kda_verify_prep(*args)  # warm the kernel cache outside the capture
        restore()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            got = kda_verify_prep(*args)
    torch.cuda.synchronize()
    unchanged = all(torch.equal(x, y) for x, y in zip(pool, before, strict=True))
    assert unchanged, "the capture itself must not run the kernel"
    graph.replay()
    torch.cuda.synchronize()
    for a, b in zip(got, want, strict=True):
        assert torch.equal(a, b)
    assert all(torch.equal(x, y) for x, y in zip(pool, pool_after_eager, strict=True))
